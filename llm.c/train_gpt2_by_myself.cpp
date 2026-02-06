/*
This file trains the GPT-2 model.
This version is the clean, minimal, reference. As such:
- it runs on CPU.
- it does not make the code too complex; it is readable.
- it does not use any processor-specific instructions, intrinsics and such.
- it _does_ use a few OpenMP pragmas because this is a large speedup at very low cost
There will be other versions of this code that specialize it and make it fast.
*/

#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>
#if defined(_MSC_VER)
#include <io.h>
#else
#include <unistd.h>
#endif
#ifdef OMP
#include <omp.h>
#endif

#include <stdarg.h> // 处理可变参数 (va_list)
#include <time.h>   // 处理日期和时间

// 在原有的 include 后面添加
#include <sys/stat.h>
#include <sys/types.h>

#define M_PI 3.14159265358979323846

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

void get_date_log_filename(char* buffer, size_t size);
// 2. 双重日志函数：既打印到屏幕，又追加到文件
void log_message(const char* filename, const char* format, ...);

void encoder_forward(float* out,
                   int* inp, float* wte, float* wpe,
                   int B, int T, int C) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            // get the index of the token at inp[b, t]
            int ix = inp[b * T + t];
            // seek to the position in wte corresponding to the token
            float* wte_ix = wte + ix * C;
            // seek to the position in wpe corresponding to the position
            float* wpe_t = wpe + t * C;
            // add the two vectors and store the result in out[b,t,:]
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

void encoder_backward(float* dwte, float* dwpe,
                      float* dout, int* inp,
                      int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            int ix = inp[b * T + t];
            float* dwte_ix = dwte + ix * C;
            float* dwpe_t = dwpe + t * C;
            for (int i = 0; i < C; i++) {
                float d = dout_bt[i];
                dwte_ix[i] += d;
                dwpe_t[i] += d;
            }
        }
    }
}

void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalize
                float o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

void layernorm_backward(float* dinp, float* dweight, float* dbias,
                        float* dout, float* inp, float* weight, float* mean, float* rstd,
                        int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            float* inp_bt = inp + b * T * C + t * C;
            float* dinp_bt = dinp + b * T * C + t * C;
            float mean_bt = mean[b * T + t];
            float rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i; // term 1
                dval -= dnorm_mean; // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt; // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

void matmul_forward(float* out,
                    float* inp, float* weight, float* bias,
                    int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_bt = out + b * T * OC + t * OC;
            float* inp_bt = inp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                float* wrow = weight + o*C;
                for (int i = 0; i < C; i++) {
                    val += inp_bt[i] * wrow[i];
                }
                out_bt[o] = val;
            }
        }
    }
}

void matmul_backward(float* dinp, float* dweight, float* dbias,
                     float* dout, float* inp, float* weight,
                     int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * OC + t * OC;
            float* dinp_bt = dinp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float* wrow = weight + o*C;
                float d = dout_bt[o];
                for (int i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
    // backward into weight/bias, parallelize over output channels OC
    #pragma omp parallel for
    for (int o = 0; o < OC; o++) {
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                float* dout_bt = dout + b * T * OC + t * OC;
                float* inp_bt = inp + b * T * C + t * C;
                float* dwrow = dweight + o*C;
                float d = dout_bt[o];
                if (dbias != NULL) { dbias[o] += d; }
                for (int i = 0; i < C; i++) {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
    }
}

void attention_forward(float* out, float* preatt, float* att,
                       float* inp,
                       int B, int T, int C, int NH) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                float maxval = -10000.0f; // TODO something better
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

void attention_backward(float* dinp, float* dpreatt, float* datt,
                        float* dout, float* inp, float* att,
                        int B, int T, int C, int NH) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;
                float* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                float* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                float* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                float* dout_bth = dout + b * T * C + t * C + h * hs;
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for (int t2 = 0; t2 <= t; t2++) {
                    for (int t3 = 0; t3 <= t; t3++) {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    float* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        // so now we have:
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale;
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                    }
                }
            }
        }
    }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward(float* out, float* inp, int N) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
#pragma float_control(precise, on, push) // On msvc /fp:fast is a lot faster, but the expf inside coshf breaks the model
__attribute__((optimize("no-finite-math-only"))) // same for gcc -Ofast
void gelu_backward(float* dinp, float* inp, float* dout, int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += local_grad * dout[i];
    }
}
#pragma float_control(pop)

void residual_forward(float* out, float* inp1, float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

void residual_backward(float* dinp1, float* dinp2, float* dout, int N) {
    for (int i = 0; i < N; i++) {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}

void softmax_forward(float* probs, float* logits, int B, int T, int V) {
    // output: probs are (B,T,V) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,V) of the unnormalized log probabilities
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // probs <- softmax(logits)
            float* logits_bt = logits + b * T * V + t * V;
            float* probs_bt = probs + b * T * V + t * V;

            // maxval is only calculated and subtracted for numerical stability
            float maxval = -10000.0f; // TODO something better
            for (int i = 0; i < V; i++) {
                if (logits_bt[i] > maxval) {
                    maxval = logits_bt[i];
                }
            }
            float sum = 0.0f;
            for (int i = 0; i < V; i++) {
                probs_bt[i] = expf(logits_bt[i] - maxval);
                sum += probs_bt[i];
            }
            for (int i = 0; i < V; i++) {
                probs_bt[i] /= sum;
            }
        }
    }
}

void crossentropy_forward(float* losses,
                          float* probs, int* targets,
                          int B, int T, int V) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,V) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // loss = -log(probs[target])
            float* probs_bt = probs + b * T * V + t * V;
            int ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

void crossentropy_softmax_backward(float* dlogits,
                           float* dlosses, float* probs, int* targets,
                           int B, int T, int V) {
    // backwards through both softmax and crossentropy
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dlogits_bt = dlogits + b * T * V + t * V;
            float* probs_bt = probs + b * T * V + t * V;
            float dloss = dlosses[b * T + t];
            int ix = targets[b * T + t];
            for (int i = 0; i < V; i++) {
                float p = probs_bt[i];
                float indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
typedef struct {
    // 嵌入层 (Embeddings) —— 数据的入口
    // 全称: Weight Token Embedding。含义: 词嵌入矩阵。
    float* wte; // (V, C)   // 维度: 词表大小 $V$ (50257) $\times$ 通道数 $C$ (768)。
    // 全称: Weight Positional Embedding。含义: 位置嵌入矩阵。
    float* wpe; // (maxT, C)    // 维度: 最大序列长度 $T$ (1024) $\times$ 通道数 $C$ (768)。

    // 2. Transformer 层 (The Layers) —— 核心计算单元
    /*
    GPT-2 有 $L$ 层（例如 12 层），每层包含两个子模块：Attention（注意力） 和 MLP（前馈网络）。
    这部分参数占据了模型 90% 以上的大小。
    */
    // A. 第一半：注意力机制 (Attention)
    // 全称: Layer Norm 1 Weight / Bias。
    // 作用: 在进入注意力层之前，对数据进行层归一化。w 是缩放（Scale），b 是平移（Shift）。
    float* ln1w; // (L, C)  维度: $L$ 层，每层有 $C$ 个参数。
    float* ln1b; // (L, C)  维度: $L$ 层，每层有 $C$ 个参数。
    // 全称: Query-Key-Value Projection Weight / Bias。
    // 作用: 注意力投影。它把输入向量同时转换成 Query（查询）、Key（键）、Value（值）三种向量。
    // 为了计算快，这三个矩阵被拼在一起存储，所以维度是 $3 \times C$。
    float* qkvw; // (L, 3*C, C)
    float* qkvb; // (L, 3*C)
    // 全称: Attention Projection Weight / Bias。
    // 作用: 输出投影。多头注意力算完后，把结果拼起来，通过这个矩阵进行特征融合，还原回 $C$ 维。
    float* attprojw; // (L, C, C)
    float* attprojb; // (L, C)

    // B. 第二半：前馈网络 (MLP)
    // 全称: Layer Norm 2 Weight / Bias。作用: 在进入 MLP 层之前的层归一化。
    float* ln2w; // (L, C)
    float* ln2b; // (L, C)
    // 全称: Fully Connected Weight / Bias。
    // 作用: 升维层。它把 $C$ 维向量投影到 $4 \times C$ 维（例如 $768 \to 3072$）。
    // 变宽是为了让模型能处理更复杂的特征。中间会经过 GELU 激活函数。
    float* fcw; // (L, 4*C, C)
    float* fcb; // (L, 4*C)
    // 全称: Fully Connected Projection Weight / Bias。
    // 作用: 降维层。把变宽的向量再压缩回 $C$ 维，以便传给下一层。
    float* fcprojw; // (L, C, 4*C)
    float* fcprojb; // (L, C)

    // 3. 输出层 (Final Output) —— 最后的整理
    // 全称: Layer Norm Final Weight / Bias。
    // 作用: 经过 12 层“折磨”后，数据分布可能偏了。在输出预测之前，最后做一次归一化。
    float* lnfw; // (C)
    float* lnfb; // (C)
} ParameterTensors;

// allocate memory for the parameters and point the individual tensors to the right places
float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes) {
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    // malloc all parameters all at once
    float* params_memory = (float*)malloc(num_parameters * sizeof(float));
    // assign all the tensors
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    float* params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

#define NUM_ACTIVATION_TENSORS 23
/*
ActivationTensors 结构体存储了模型在前向传播 (Forward Pass) 过程中产生的所有中间结果（Activations）。
为什么要存这些？
这主要是为了反向传播 (Backward Pass)。当我们要计算梯度时，根据链式法则，我们不仅需要当前的梯度，还需要当时的输入值。
例如，$y = x^2$ 的导数是 $2x$，如果你没把 $x$ 存下来，反向传播时就没法算出梯度了。

这些变量按照数据在 GPT-2 内部流动的顺序，可
以分为  1输入阶段 、 2Transformer  层内部（Attention + MLP）、 3输出阶段  三大块。
*/
typedef struct {
    // 1. 输入阶段 (Embedding)
    // 含义: 编码后的输入。来源:它是 Token Embedding (wte) 和 Positional Embedding (wpe) 相加后的结果。
    // 作用: 这是第 0 层 Transformer Block 的原始输入。
    float* encoded; // (B, T, C)

    // 2. Transformer 层内部 (The Blocks)
    // GPT-2 有 $L$ 层，每一层内部又分为“注意力”和“前馈网络”两半。
    // 因为每一层的数据都需要保存，所以这些变量的维度都带有一个 $L$。

    // A. 第一半：注意力机制 (Attention Block)
    // 含义: LayerNorm 1 的输出。作用: 这是进入 Attention 层之前的数据。
    float* ln1; // (L, B, T, C)
    // 含义: LayerNorm 计算出的均值 ($\mu$) 和标准差的倒数 ($1/\sigma$)。
    // 作用: 反向传播算 LayerNorm 的梯度时必须用到这两个统计量。
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    // 含义: Query, Key, Value 的合体。
    // 来源: 输入向量经过 c_attn 投影层后的结果。
    // 作用: 这里包含了计算注意力所需的所有原材料。
    float* qkv; // (L, B, T, 3*C)
    // 含义: 注意力层的输出 (Y)。
    // 来源: $Att \times V$ 的结果。这时所有头 (Heads) 的结果已经拼在了一起。
    float* atty; // (L, B, T, C)
    // 含义: 注意力分数 (Attention Scores)。
    // 来源: $Q \times K^T$ 的结果（还没有做 Softmax）。
    // 维度: $T \times T$ 的矩阵，代表每个字对其他所有字的“关注度”。
    float* preatt; // (L, B, NH, T, T)
    // 含义: 注意力权重 (Attention Probabilities)。
    // 来源: preatt 经过 Softmax 后的结果（概率和为 1）。
    float* att; // (L, B, NH, T, T)
    // 含义: 投影后的输出。
    // 来源: atty 经过线性层 (c_proj) 后的结果。
    float* attproj; // (L, B, T, C)
    // 含义: 第一个残差连接点。
    // 公式: residual2 = Input + attproj。
    // 作用: 它是 Attention 模块结束后的最终结果，也是 MLP 模块的输入。
    float* residual2; // (L, B, T, C)

    // B. 第二半：前馈网络 (MLP Block)
    // 含义: LayerNorm 2 的输出。
    // 作用: 进入 MLP 之前的数据。同样也有对应的 ln2_mean 和 ln2_rstd 用于存统计量。
    float* ln2; // (L, B, T, C)
    // 含义: 第 2 个层归一化 (LayerNorm 2) 的均值。
    // 位置: 在每一个 Transformer 层内部，进入 MLP（前馈网络） 之前的那次归一化。
    // 维度: (L, B, T)。因为模型有 $L$ 层，每一层都有自己的 MLP 模块，也就都有自己的 ln2 操作。
    // 所以需要记录 $L$ 层中所有位置的均值。
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    // 含义: FC Hidden (升维层输出)。来源: 经过第一个全连接层，维度从 $C$ 变大到 $4C$。
    float* fch; // (L, B, T, 4*C)
    // 含义: 激活后的中间层。来源: fch 经过 GELU 激活函数后的结果。
    float* fch_gelu; // (L, B, T, 4*C)
    // 含义: MLP 的最终输出。来源: 维度从 $4C$ 缩回 $C$。
    float* fcproj; // (L, B, T, C)
    // 含义: 第二个残差连接点 (Block Output)。
    // 公式: residual3 = residual2 + fcproj。作用: 这一层的最终输出，同时也是下一层的输入。
    float* residual3; // (L, B, T, C)

    // 3. 输出阶段 (Final Output)
    // 含义: 最终 LayerNorm (Final)。
    // 作用: 整个 Transformer 最后的标准化输出。也有对应的 lnf_mean 和 lnf_rstd。
    float* lnf; // (B, T, C)
    // 含义: 最终层归一化 (Final LayerNorm) 的均值。
    // 位置: 模型所有层跑完后，在输出 logits 之前，有一个 lnf (LayerNorm Final)。
    // 作用: 供 lnf 的反向传播使用。
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)
    // 含义: 逻辑值 (Logits)。来源: lnf 乘以 词嵌入矩阵 (wte) 的转置。
    // 维度: 这里的最后一维变成了 $V$ (Vocab Size, 50257)。它代表模型对下一个词是词表中每一个词的“打分”。
    float* logits; // (B, T, V)
    // 含义: 预测概率。来源: logits 经过 Softmax。所有词的概率加起来等于 1。
    float* probs; // (B, T, V)
    // 含义: 损失值。
    // 来源: 计算预测概率与真实标签 (targets) 之间的交叉熵。
    // 每个位置都有一个 Loss，最后会被平均成一个数 (mean_loss)。
    float* losses; // (B, T)
} ActivationTensors;

float* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes) {
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        num_activations += act_sizes[i];
    }
    float* acts_memory = (float*)malloc(num_activations * sizeof(float));
    float** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses
    };
    float* acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768, 嵌入向量的维度
} GPT2Config;

/*
    GPT2 结构体是整个代码中最核心的数据结构，它封装了一个 GPT-2 模型训练所需的所有数据，
    包括模型的“骨架”（配置）、“肉体”（权重参数）、“短期记忆”（激活值）以及“学习经验”（梯度和优化器状态）。
*/
typedef struct {
    // 将这个结构体看作一个工厂的总控台，
    // 它管理着模型训练的五个主要区域：1配置区、2参数区、3梯度区、4优化器区 和 5运行时状态区。

    // 配置区
    /*
    作用：存储模型的“设计图纸”或超参数。
    内容：包含最大序列长度 (max_seq_len)、词表大小 (vocab_size)、层数 (num_layers)、
    头数 (num_heads) 和维度 (channels) 等。这些决定了模型的大小和结构。
    */
    GPT2Config config;

    // the weights (parameters) of the model, and their sizes
    // 参数区 (Weights/Parameters) —— 模型的“知识”
    /*
    这是一个结构体 (ParameterTensors)，里面包含了一堆指针（如 wte, wpe, ln1w 等）。
    这些指针并不拥有内存，而是指向 params_memory 中对应的位置。
    这样可以方便地通过名字（如 model.params.wte）来访问权重，而底层的内存管理依然是扁平高效的。
    */
    ParameterTensors params;    
    size_t param_sizes[NUM_PARAMETER_TENSORS];  // 记录每个权重张量的大小（元素个数），用于计算偏移量。
    float* params_memory;   // 一个巨大的、连续的浮点数数组。它是所有权重的实际物理存储位置。
    size_t num_parameters;  // 权重的总数量（例如 1.24 亿个 float）。

    // 梯度区 (Gradients)
    // gradients of the weights
    ParameterTensors grads; // 里面的指针指向 grads_memory 的对应位置。grads.wte 存储的就是 params.wte 的梯度。
    float* grads_memory;    // 结构和 params_memory 完全一样大，也是一块连续内存。

    // buffers for the AdamW optimizer
    // 优化器状态区
    float* m_memory;    // 一阶动量 (Momentum)
    float* v_memory;    // 二阶动量 (Variance)

    // the activations of the model, and their sizes
    // 激活值区 (Activations) —— 模型的“短期记忆”
    // 在前向传播（Forward）中产生的中间结果，必须保存下来，供反向传播（Backward）使用。
    ActivationTensors acts; // 类似于 params，包含指向 acts_memory 的指针（如 encoded, qkv, att 等）。
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float* acts_memory; // 存储所有层的输出、Attention 的中间结果、LayerNorm 的均值方差等。
    size_t num_activations;

    // gradients of the activations
    // 激活梯度区 (Gradients of Activations) —— 反向传播的“草稿纸”
    // 在反向传播时，误差不仅要传给权重，还要传给上一层的激活值。
    // 这里存储的就是对激活值的梯度（例如 $\frac{\partial Loss}{\partial \text{LayerNorm\_Output}}$）。
    ActivationTensors grads_acts;
    float* grads_acts_memory;

    // other run state configuration
    // 运行时状态区 (Run State) —— 当前的“工况”
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    int* inputs; // the input tokens for the current forward pass
    int* targets; // the target tokens for the current forward pass
    float mean_loss; // after a forward pass with targets, will be populated with the mean loss
} GPT2;

void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path, const char* log_filename) {

    // read in model from a checkpoint file
    // "rb" 读取二进制
    FILE *model_file = fopen(checkpoint_path, "rb");
    if (model_file == NULL) { printf("Error opening model file\n"); exit(1); }
    // ... 错误检查 ...
    int model_header[256];
    fread(model_header, sizeof(int), 256, model_file);  // 读取前 256 个整数
    if (model_header[0] != 20260123) { printf("Bad magic model file"); exit(1); }   // Magic Number 校验
    if (model_header[1] != 1) { printf("Bad version in model file"); exit(1); }     // 版本校验

    // read in hyperparameters
    int maxT, V, L, NH, C;
    model->config.max_seq_len = maxT = model_header[2]; // 最大序列长度 (如 1024)
    model->config.vocab_size = V = model_header[3];     // 词表大小 (如 50257)
    // ... 以及层数 L, 头数 NH, 维度 C
    model->config.num_layers = L = model_header[4];
    model->config.num_heads = NH = model_header[5];
    model->config.channels = C = model_header[6];
    // printf("[GPT-2]\n");
    // printf("max_seq_len: %d\n", maxT);
    // printf("vocab_size: %d\n", V);
    // printf("num_layers: %d\n", L);
    // printf("num_heads: %d\n", NH);
    // printf("channels: %d\n", C);
    log_message(log_filename, "[GPT-2]\n");
    log_message(log_filename, "max_seq_len: %d\n", maxT);
    log_message(log_filename, "vocab_size: %d\n", V);
    log_message(log_filename, "num_layers: %d\n", L);
    log_message(log_filename, "num_heads: %d\n", NH);
    log_message(log_filename, "channels: %d\n", C);



    // allocate space for all the parameters and read them in
    // 计算参数尺寸, 计算每个张量（Tensor）具体有多少个浮点数
    model->param_sizes[0] = V * C; // wte   // wte: Token Embedding (词表大小 x 维度)
    model->param_sizes[1] = maxT * C; // wpe    // wpe: Positional Embedding (长度 x 维度)
    model->param_sizes[2] = L * C; // ln1w
    model->param_sizes[3] = L * C; // ln1b
    model->param_sizes[4] = L * (3 * C) * C; // qkvw    // qkvw: Attention 权重 (3倍是因为 Q,K,V 在一起)
    model->param_sizes[5] = L * (3 * C); // qkvb
    model->param_sizes[6] = L * C * C; // attprojw
    model->param_sizes[7] = L * C; // attprojb
    model->param_sizes[8] = L * C; // ln2w
    model->param_sizes[9] = L * C; // ln2b
    model->param_sizes[10] = L * (4 * C) * C; // fcw    // fcw: MLP 升维层 (通常是 4倍维度)
    model->param_sizes[11] = L * (4 * C); // fcb
    model->param_sizes[12] = L * C * (4 * C); // fcprojw
    model->param_sizes[13] = L * C; // fcprojb
    model->param_sizes[14] = C; // lnfw
    model->param_sizes[15] = C; // lnfb

    // count the number of parameters
    // 统计总参数量
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    // printf("num_parameters: %zu\n", num_parameters);
    log_message(log_filename, "num_parameters: %zu\n", num_parameters);
    model->num_parameters = num_parameters;

    // read in all the parameters from file
    // 内存分配与加载 (Allocation & Loading) —— 核心设计
    // 这是该代码最高效、最值得学习的地方：扁平化内存管理。

    // 1. 分配一块巨大的连续内存，并设置指针偏移
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    // 2. 一次性从磁盘把所有权重读入这块内存
    fread(model->params_memory, sizeof(float), num_parameters, model_file);
    fclose(model_file);

    // other inits
    // 初始化其他状态 (State Init)
    model->acts_memory = NULL;  // 激活值内存先不分，Forward 时再懒加载, 前向传播的中间计算结果（Activations）的存放地
    model->grads_memory = NULL; // 梯度内存先不分，Backward 时再分
    model->m_memory = NULL; // 优化器状态（AdamW 的一阶动量 $m$ 和二阶动量 $v$）, 如果是推理模式，完全不需要这两块大内存。
    model->v_memory = NULL; // 优化器状态（AdamW 的一阶动量 $m$ 和二阶动量 $v$）, 如果是推理模式，完全不需要这两块大内存。
    model->grads_acts_memory = NULL;    // 激活值的梯度, 反向传播时的临时草稿纸。同样，推理模式不需要。
    model->inputs = NULL;   // 用于在 GPU/CPU 内部缓存输入数据的 buffer。
    model->targets = NULL;  // 用于在 GPU/CPU 内部缓存输入数据的 buffer。
    model->batch_size = 0;  // 记录当前内存是为了多大的 $B$ 和 $T$ 分配的。
    model->seq_len = 0;     // 记录当前内存是为了多大的 $B$ 和 $T$ 分配的。
    model->mean_loss = -1.0f; // -1.0f will designate no loss // 标记还没有计算过 Loss
}

void gpt2_forward(GPT2 *model, int* inputs, int* targets, int B, int T) {
    // targets are optional and could be NULL

    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(1);
    }

    // convenience parameters
    int V = model->config.vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;

    // validate inputs, all indices must be in the range [0, V)
    for(int i = 0; i < B * T; i++) {
        assert(0 <= inputs[i] && inputs[i] < V);
        if (targets != NULL) {
            assert(0 <= targets[i] && targets[i] < V);
        }
    }

    // allocate space for all the activations if needed (done here, lazily)
    if(model->acts_memory == NULL) {
        // record the current B,T as well
        model->batch_size = B;
        model->seq_len = T;
        // and now allocate the space
        model->act_sizes[0] = B * T * C; // encoded
        model->act_sizes[1] = L * B * T * C; // ln1
        model->act_sizes[2] = L * B * T;  // ln1_mean
        model->act_sizes[3] = L * B * T;  // ln1_rstd
        model->act_sizes[4] = L * B * T * 3*C; // qkv
        model->act_sizes[5] = L * B * T * C;  // atty
        model->act_sizes[6] = L * B * NH * T * T;  // preatt
        model->act_sizes[7] = L * B * NH * T * T;  // att
        model->act_sizes[8] = L * B * T * C; // attproj
        model->act_sizes[9] = L * B * T * C; // residual2
        model->act_sizes[10] = L * B * T * C; // ln2
        model->act_sizes[11] = L * B * T; // ln2_mean
        model->act_sizes[12] = L * B * T; // ln2_rstd
        model->act_sizes[13] = L * B * T * 4*C; // fch
        model->act_sizes[14] = L * B * T * 4*C; // fch_gelu
        model->act_sizes[15] = L * B * T * C; // fcproj
        model->act_sizes[16] = L * B * T * C; // residual3
        model->act_sizes[17] = B * T * C; // lnf
        model->act_sizes[18] = B * T; // lnf_mean
        model->act_sizes[19] = B * T; // lnf_rstd
        model->act_sizes[20] = B * T * V; // logits
        model->act_sizes[21] = B * T * V; // probs
        model->act_sizes[22] = B * T; // losses
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }
        printf("num_activations: %zu\n", num_activations);
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        // also create memory for caching inputs and targets
        model->inputs = (int*)malloc(B * T * sizeof(int));
        model->targets = (int*)malloc(B * T * sizeof(int)); // might be unused if we never have targets but it's small
    } else {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        if (B != model->batch_size || T != model->seq_len) {
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, B, T);
            exit(EXIT_FAILURE);
        }
    }

    // cache the inputs/targets
    memcpy(model->inputs, inputs, B * T * sizeof(int));
    if (targets != NULL) {
        memcpy(model->targets, targets, B * T * sizeof(int));
    }

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    float* residual;
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]
    for (int l = 0; l < L; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_ln1b = params.ln1b + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_qkvb = params.qkvb + l * 3*C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_attprojb = params.attprojb + l * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_ln2b = params.ln2b + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcb = params.fcb + l * 4*C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        float* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkv = acts.qkv + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_preatt = acts.preatt + l * B * NH * T * T;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_attproj = acts.attproj + l * B * T * C;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        float* l_fcproj = acts.fcproj + l * B * T * C;
        float* l_residual3 = acts.residual3 + l * B * T * C;

        // now do the forward pass
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    }
    residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, V);
    softmax_forward(acts.probs, acts.logits, B, T, V);

    // also forward the cross-entropy loss function if we have the targets
    if (targets != NULL) {
        crossentropy_forward(model->acts.losses, model->acts.probs, targets, B, T, V);
        // for convenience also evaluate the mean loss
        float mean_loss = 0.0f;
        for (int i=0; i<B*T; i++) { mean_loss += model->acts.losses[i]; }
        mean_loss /= B*T;
        model->mean_loss = mean_loss;
    } else {
        // if we don't have targets, we don't have a loss
        model->mean_loss = -1.0f;
    }
}

void gpt2_zero_grad(GPT2 *model) {
    if(model->grads_memory != NULL) { memset(model->grads_memory, 0, model->num_parameters * sizeof(float)); }
    if(model->grads_acts_memory != NULL) { memset(model->grads_acts_memory, 0, model->num_activations * sizeof(float)); }
}

void gpt2_backward(GPT2 *model) {

    // double check we forwarded previously, with targets
    if (model->mean_loss == -1.0f) {
        printf("Error: must forward with targets before backward\n");
        exit(1);
    }

    // lazily allocate the memory for gradients of the weights and activations, if needed
    if (model->grads_memory == NULL) {
        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes);
        model->grads_acts_memory = malloc_and_point_activations(&model->grads_acts, model->act_sizes);
        gpt2_zero_grad(model);
    }

    // convenience shortcuts
    int B = model->batch_size;
    int T = model->seq_len;
    int V = model->config.vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;

    // backward pass: go in the reverse order of the forward pass, and call backward() functions
    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    ActivationTensors grads_acts = model->grads_acts;

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // technically this is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    float dloss_mean = 1.0f / (B*T);
    for (int i = 0; i < B*T; i++) { grads_acts.losses[i] = dloss_mean; }

    crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, model->targets, B, T, V);
    matmul_backward(grads_acts.lnf, grads.wte, NULL, grads_acts.logits, acts.lnf, params.wte, B, T, C, V);
    float* residual = acts.residual3 + (L-1) * B * T * C; // last layer's residual
    float* dresidual = grads_acts.residual3 + (L-1) * B * T * C; // write to last layer's residual
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);

    for (int l = L-1; l >= 0; l--) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;
        dresidual = l == 0 ? grads_acts.encoded : grads_acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        // get the pointers of the gradients of the weights for this layer
        float* dl_ln1w = grads.ln1w + l * C;
        float* dl_ln1b = grads.ln1b + l * C;
        float* dl_qkvw = grads.qkvw + l * 3*C * C;
        float* dl_qkvb = grads.qkvb + l * 3*C;
        float* dl_attprojw = grads.attprojw + l * C * C;
        float* dl_attprojb = grads.attprojb + l * C;
        float* dl_ln2w = grads.ln2w + l * C;
        float* dl_ln2b = grads.ln2b + l * C;
        float* dl_fcw = grads.fcw + l * 4*C * C;
        float* dl_fcb = grads.fcb + l * 4*C;
        float* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        float* dl_fcprojb = grads.fcprojb + l * C;
        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkv = acts.qkv + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        // get the pointers of the gradients of the activations for this layer
        float* dl_ln1 = grads_acts.ln1 + l * B * T * C;
        float* dl_qkv = grads_acts.qkv + l * B * T * 3*C;
        float* dl_atty = grads_acts.atty + l * B * T * C;
        float* dl_preatt = grads_acts.preatt + l * B * NH * T * T;
        float* dl_att = grads_acts.att + l * B * NH * T * T;
        float* dl_attproj = grads_acts.attproj + l * B * T * C;
        float* dl_residual2 = grads_acts.residual2 + l * B * T * C;
        float* dl_ln2 = grads_acts.ln2 + l * B * T * C;
        float* dl_fch = grads_acts.fch + l * B * T * 4*C;
        float* dl_fch_gelu = grads_acts.fch_gelu + l * B * T * 4*C;
        float* dl_fcproj = grads_acts.fcproj + l * B * T * C;
        float* dl_residual3 = grads_acts.residual3 + l * B * T * C;

        // backprop this layer
        residual_backward(dl_residual2, dl_fcproj, dl_residual3, B*T*C);
        matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4*C, C);
        gelu_backward(dl_fch, l_fch, dl_fch_gelu, B*T*4*C);
        matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4*C);
        layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
        residual_backward(dresidual, dl_attproj, dl_residual2, B*T*C);
        matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C);
        attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
        matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3*C);
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
    }
    encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, model->inputs, B, T, C);
}

void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    // lazily allocate the memory for m_memory and v_memory
    if (model->m_memory == NULL) {
        model->m_memory = (float*)calloc(model->num_parameters, sizeof(float));
        model->v_memory = (float*)calloc(model->num_parameters, sizeof(float));
    }

    for (int i = 0; i < model->num_parameters; i++) {
        float param = model->params_memory[i];
        float grad = model->grads_memory[i];

        // update the first moment (momentum)
        float m = beta1 * model->m_memory[i] + (1.0f - beta1) * grad;
        // update the second moment (RMSprop)
        float v = beta2 * model->v_memory[i] + (1.0f - beta2) * grad * grad;
        // bias-correct both moments
        float m_hat = m / (1.0f - powf(beta1, t));
        float v_hat = v / (1.0f - powf(beta2, t));

        // update
        model->m_memory[i] = m;
        model->v_memory[i] = v;
        model->params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
}

void gpt2_free(GPT2 *model) {
    free(model->params_memory);
    free(model->grads_memory);
    free(model->m_memory);
    free(model->v_memory);
    free(model->acts_memory);
    free(model->grads_acts_memory);
    free(model->inputs);
    free(model->targets);
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.c), we'll skip the int main below

// ----------------------------------------------------------------------------
// data loader lite
// returns random batches of data from a file of integers

typedef struct {
    // hyperparameters
    int B; // batch size
    int T; // sequence length
    // input handling and its state
    FILE* tokens_file;
    long file_size;
    long current_position;
    // output memory
    int* batch;
    int* inputs;
    int* targets;
    // convenience variables
    int num_batches;
} DataLoader;

void dataloader_init(DataLoader *loader, const char* filename, int B, int T) {
    loader->B = B;
    loader->T = T;

    // open the input file for reading
    loader->tokens_file = fopen(filename, "rb");
    if (loader->tokens_file == NULL) {
        printf("Error opening tokens file\n");
        exit(1);
    }

    // determine the file size
    fseek(loader->tokens_file, 0, SEEK_END);
    loader->file_size = ftell(loader->tokens_file);
    fseek(loader->tokens_file, 0, SEEK_SET);
    if (loader->file_size < (B * T + 1) * sizeof(int)) {
        printf("Error: file size is too small for the batch size and sequence length\n");
        exit(1);
    }
    loader->current_position = 0; // start at the beginning

    // allocate space for B*T + 1 integers to store the inputs and targets
    loader->batch = (int*) malloc((B * T + 1) * sizeof(int));
    loader->inputs = loader->batch;
    loader->targets = loader->batch + 1; // targets are shifted by one
    loader->num_batches = loader->file_size / (B * T * sizeof(int));
}

void dataloader_reset(DataLoader *loader) {
    loader->current_position = 0;
}

void dataloader_next_batch(DataLoader *loader) {
    int B = loader->B;
    int T = loader->T;
    // if we are at the end of the file, loop back to the beginning
    if (loader->current_position + (B*T+1) * sizeof(int) > loader->file_size) {
        loader->current_position = 0;
    }
    // read the B*T+1 integers from the file into batch
    fseek(loader->tokens_file, loader->current_position, SEEK_SET);
    fread(loader->batch, sizeof(int), B*T+1, loader->tokens_file);
    // advance the current position by B*T integers
    loader->current_position += B*T * sizeof(int);
}

void dataloader_free(DataLoader *loader) {
    fclose(loader->tokens_file);
    free(loader->batch);
}

// ----------------------------------------------------------------------------
// sampler

// the GPT-2 end-of-text token id
#define GPT2_EOT 1

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// Tokenizer (only supports decoding)

typedef struct {
    uint32_t vocab_size;
    char **token_table;
    int init_ok;
} Tokenizer;

void safe_printf(const char *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // weird byte, don't print it
        }
    }
    printf("%s", piece);
}

void tokenizer_init(Tokenizer *tokenizer, const char *filename) {
    // // 1. 打开文件 (必须是二进制模式 "rb")
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        // try to be more helpful as we just added this feature, erase later
        printf("---\n");
        printf("WARNING: Failed to open the tokenizer file %s\n", filename);
        printf("The Tokenizer is a new feature added April 14 2024.\n");
        printf("Re-run `python train_gpt2.py` to write it\n");
        printf("---\n");
        tokenizer->init_ok = 0;
        return;
    }
    // read in the header
    // // 2. 读取文件头 (Header) —— 固定 1024 字节
    uint32_t header[256];
    fread(header, sizeof(uint32_t), 256, file);
    // // 3. 校验魔数和版本 (确保没有读错文件)
    assert(header[0] == 20260123);  // Magic Number
    assert(header[1] == 1);         // Version
    tokenizer->vocab_size = header[2];  // 拿到词表大小 (比如 50257)

    // read in all the tokens
    // 4. 循环读取每一个 Token (Body)
    unsigned char length;   // 这是一个 1 字节的整数 (0~255)
    // 申请指针数组
    tokenizer->token_table = (char **)malloc(tokenizer->vocab_size * sizeof(char *));
    for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        // A. 先读 1 个字节：代表当前这个词有多长
        fread(&length, sizeof(unsigned char), 1, file);
        assert(length > 0); // every token should be at least one character
        // B. 申请内存：长度 + 1 (留给 \0 结束符)
        char *token_bytes = (char *)malloc(length + 1);
        // C. 再读 N 个字节：这就是词的具体内容 (UTF-8 编码)
        fread(token_bytes, sizeof(char), length, file);
        token_bytes[length] = '\0';  // Add null terminator for printing
        tokenizer->token_table[i] = token_bytes;    // 存入表中
    }
    // cleanups
    fclose(file);
    tokenizer->init_ok = 1;
}

const char *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    if (tokenizer->init_ok == 0) {
        return NULL;
    }
    if (token_id < tokenizer->vocab_size) {
        return tokenizer->token_table[token_id];
    } else {
        printf("invalid token id %d!\n", token_id);
        return NULL;
    }
}

void tokenizer_free(Tokenizer *tokenizer) {
    if (tokenizer->init_ok) {
        for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            free(tokenizer->token_table[i]);
        }
        free(tokenizer->token_table);
    }
}

// 把它放在 main 函数外面, 保存权重
void gpt2_save(GPT2 *model, const char* filename) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        printf("Error: 无法打开文件 %s 进行写入\n", filename);
        return;
    }

    // 1. 写入 Header (256个 int)
    // 这里的顺序必须和 Python 里的 write_model 一模一样！
    int header[256] = {0};
    header[0] = 20260123; // Magic
    header[1] = 1;        // Version
    header[2] = model->config.max_seq_len;
    header[3] = model->config.vocab_size;
    header[4] = model->config.num_layers;
    header[5] = model->config.num_heads;
    header[6] = model->config.channels;
    fwrite(header, sizeof(int), 256, file);

    // 2. 写入所有参数 (Weights)
    // 假设 model->params_memory 是所有参数的连续内存指针
    fwrite(model->params_memory, sizeof(float), model->num_parameters, file);

    fclose(file);
    printf("已保存模型到: %s\n", filename);
}

// --- 辅助函数：打印帮助信息 ---
void print_usage() {
    printf("Usage: train_gpt2_by_myself [options]\n");
    printf("Options:\n");
    printf("  -i <path>   Input model bin file (default: output_pre_model/gpt2_init.bin)\n");
    printf("  -j <path>   Training data bin file (default: output_tokenizer/train.bin)\n");
    printf("  -k <path>   Validation data bin file (default: output_tokenizer/val.bin)\n");
    printf("  -z <path>   Tokenizer bin file (default: output_tokenizer/threebody_tokenizer.bin)\n");
    // 【修改】明确说明这里只是文件名
    printf("  -o <name>   Output model filename (default: gpt2_novel.bin). Will be saved in output_model/<timestamp>/\n");
    printf("  -t <int>    Sequence length / context size (default: 64)\n");
    printf("  -b <int>    Batch size (default: 4)\n");
    printf("  -n <int>    Total number of training steps (default: 40)\n");
    printf("  -e <int>    Validate every N steps (default: 10)\n");
    printf("  -s <int>    Save checkpoint every N steps (0 = disabled, default: 0)\n");
    printf("  -l <float>  Learning rate (default: 1e-4)\n");
    exit(1);
}

// 1. 获取带日期的文件名，例如: "train_log_2026-01-28.txt"
void get_date_log_filename(char* buffer, size_t size) {
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    // 格式化: train_log_YYYY-MM-DD.txt
    strftime(buffer, size, "train_log_%Y-%m-%d.txt", t);
}

// 2. 双重日志函数：既打印到屏幕，又追加到文件
void log_message(const char* filename, const char* format, ...) {
    // --- A. 打印到屏幕 (Stdout) ---
    va_list args_console;
    va_start(args_console, format);
    vprintf(format, args_console); // vprintf 专门处理 va_list
    va_end(args_console);

    // --- B. 追加到文件 (File) ---
    if (filename && filename[0] != '\0') {
        FILE* fp = fopen(filename, "a"); // "a" = append (追加模式)
        if (fp) {
            // 可选：在文件里每行前面加具体时间 (HH:MM:SS)，方便排查
            // time_t now = time(NULL);
            // char time_buf[32];
            // strftime(time_buf, sizeof(time_buf), "[%H:%M:%S] ", localtime(&now));
            // fprintf(fp, "%s", time_buf);

            va_list args_file;
            va_start(args_file, format);
            vfprintf(fp, format, args_file); // vfprintf 专门写入文件
            va_end(args_file);
            
            fclose(fp); // 写完立刻关闭，防止程序崩溃丢失日志
        }
    }
}

// ----------------------------------------------------------------------------
// main training loop
int main(int argc, char *argv[]) {
    // 1. 设置默认参数 (Default Parameters)
    char *input_model_path = (char*)"output_pre_model/gpt2_init.bin";
    char *train_data_path  = (char*)"output_tokenizer/train.bin";
    char *val_data_path    = (char*)"output_tokenizer/val.bin";
    char *tokenizer_path   = (char*)"output_tokenizer/threebody_tokenizer.bin";
    // char *output_model_path= (char*)"output_model/gpt2_novel.bin";

    // 【修改 1】这里只存文件名，不再包含路径
    char *output_model_name = (char*)"gpt2_novel.bin"; 
    // 【修改 2】固定基础输出目录
    const char *base_output_dir = "output_model";

    int T = 64;      // Sequence length
    int B = 4;       // Batch size
    int num_steps = 40;     // 训练步数
    int val_every = 10;     // 验证的间隔步数
    int save_every = 0;
    float learning_rate = 1e-4f;

    // 2. 解析命令行参数 (Command Line Parsing)
    for (int i = 1; i < argc; i++) {
        if (i + 1 >= argc) { print_usage(); } // 参数不完整，打印帮助
        
        if      (strcmp(argv[i], "-i") == 0) { input_model_path = argv[++i]; }
        else if (strcmp(argv[i], "-j") == 0) { train_data_path = argv[++i]; }
        else if (strcmp(argv[i], "-k") == 0) { val_data_path = argv[++i]; }
        else if (strcmp(argv[i], "-z") == 0) { tokenizer_path = argv[++i]; }
        // 【修改】将 -o 解析为文件名
        else if (strcmp(argv[i], "-o") == 0) { output_model_name = argv[++i]; }
        else if (strcmp(argv[i], "-t") == 0) { T = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-b") == 0) { B = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-n") == 0) { num_steps = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-e") == 0) { val_every = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-s") == 0) { save_every = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-l") == 0) { learning_rate = atof(argv[++i]); }
        else {
            printf("Unknown option: %s\n", argv[i]);
            print_usage();
        }
    }

    // === 【简化逻辑】创建时间戳目录 ===

    // A. 生成时间字符串 (例如: 2026-02-06_12-30-00)
    time_t now = time(NULL);
    struct tm *t_struct = localtime(&now);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d_%H-%M-%S", t_struct);

    // B. 拼接完整目录路径: output_model/时间戳
    char session_dir[512];
    snprintf(session_dir, sizeof(session_dir), "%s/%s", base_output_dir, timestamp);

    // C. 创建目录
    char mkdir_cmd[512];
    snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", session_dir);
    if (system(mkdir_cmd) != 0) {
        printf("Error: Failed to create dir %s\n", session_dir);
        exit(1);
    }
    printf("Created session dir: %s\n", session_dir);

    // // === 【新增 1】生成日志文件名 ===
    // char log_filename[64];
    // get_date_log_filename(log_filename, sizeof(log_filename));
    // printf("Logging to: %s\n", log_filename);

    // D. 拼接最终文件路径
    // 1. 日志路径
    char log_filename[512];
    snprintf(log_filename, sizeof(log_filename), "%s/train_log.txt", session_dir);
    
    // 2. 模型路径
    char final_model_path[512];
    snprintf(final_model_path, sizeof(final_model_path), "%s/%s", session_dir, output_model_name);

    // 打印当前配置，防止跑错
    // printf("--- Config ---\n");
    // printf("Model: %s -> %s\n", input_model_path, output_model_path);
    // printf("Data : %s (Train), %s (Val)\n", train_data_path, val_data_path);
    // printf("Dims : B=%d, T=%d\n", B, T);
    // printf("Steps: %d (Val every %d)\n", num_steps, val_every);
    // printf("----------------\n");
    log_message(log_filename, "--- Config ---\n");
    log_message(log_filename, "Logging to: %s\n", log_filename); // 把日志文件名也记进去
    log_message(log_filename, "Model: %s -> %s\n", input_model_path, final_model_path);
    log_message(log_filename, "Data : %s (Train), %s (Val)\n", train_data_path, val_data_path);
    log_message(log_filename, "Dims : B=%d, T=%d\n", B, T);
    log_message(log_filename, "Steps: %d (Val every %d)\n", num_steps, val_every);
    log_message(log_filename, "----------------\n");


    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, input_model_path, log_filename);

    // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    const char* tiny_stories_train = "data/TinyStories_train.bin";
    const char* tiny_stories_val = "data/TinyStories_val.bin";
    const char* tiny_shakespeare_train = train_data_path;
    const char* tiny_shakespeare_val = val_data_path;

    // TODO: use std::filesystem
#if defined(_MSC_VER)
    const char* train_tokens = _access(tiny_shakespeare_train, 0) != -1 ? tiny_shakespeare_train : tiny_stories_train;
    const char* val_tokens = _access(tiny_shakespeare_val, 0) != -1 ? tiny_shakespeare_val : tiny_stories_val;
#else
    const char* train_tokens = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
    const char* val_tokens = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;
#endif
    // int B = 4; // batch size 4 (i.e. 4 independent token sequences will be trained on)
    // int T = 64; // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
    // 3. 初始化搬运工 (DataLoader Init)
    DataLoader train_loader;
    /*
    这里调用了 dataloader_init 函数，它在后台做了很多脏活累活：
    1打开文件 (fopen)：拿到文件句柄。
    2计算大小 (fseek / ftell)：看看文件总共有多大。
    3计算批次：算出这个文件够我们训练多少步（num_batches）。
        公式：文件总字节数 / (B * T * sizeof(int))。
    4申请缓存：malloc 一块大小为 B * T + 1 的内存，用来暂存从硬盘读出来的数据。
        为什么是 +1？ 因为预测下一个字需要错位（输入 1~64，预测 2~65），所以实际读取要多读一个。
    */
    dataloader_init(&train_loader, train_tokens, B, T);
    printf("train dataset num_batches: %d\n", train_loader.num_batches);
    DataLoader val_loader;
    dataloader_init(&val_loader, val_tokens, B, T);
    printf("val dataset num_batches: %d\n", val_loader.num_batches);
    // 解释：这里设为 5，意味着每次验证时，会从 val_loader 里随机抽 5 个 Batch 的数据来算 Loss，
    // 算出的平均分作为验证集 Loss。
    int val_num_batches = 5;

    // build the Tokenizer
    // 初始化“翻译官”（Tokenizer）
    Tokenizer tokenizer;
    // gpt2_tokenizer.bin 是什么？
    // 它是词汇表文件。里面存储了 50257 个词的具体内容（比如 "apple", "run", "The" 等）。
    tokenizer_init(&tokenizer, tokenizer_path);

    // some memory for generating samples from the model
    // 这段代码是为模型推理（Inference）/生成文本做准备的。
    // 在训练循环中，除了不断地反向传播更新权重外，我们还希望偶尔停下来，让模型“写一段话”给我们看看，
    // 以此直观地评估它是不是学到了东西。这一块代码就是在为这个“才艺展示”环节准备工具。

    // 这是伪随机数生成器 (Pseudo-Random Number Generator) 的初始状态，俗称“种子 (Seed)”。
    unsigned long long rng_state = 1337;
    // 生成缓冲区的内存分配 (Buffer Allocation), 用来存放模型生成的 Token ID。
    int* gen_tokens = (int*)malloc(B * T * sizeof(int));
    // 生成长度限制 (Generation Length)
    const int genT = 64; // number of steps of inference we will do

    // train
    // 完整包含了深度学习训练中最重要的三个环节：
    // 1验证 (Validation)、  2推理/采样 (Inference)、  3反向传播/学习 (Training Step)。
    struct timespec start, end;
    for (int step = 0; step <= num_steps; step++) {

        // once in a while estimate the validation loss
        // 1. 验证 (Validation) —— “期中考试”
        if (step % 10 == 0) {   // 每 10 步进行一次
            float val_loss = 0.0f;
            // 1. 重置验证集加载器，确保每次都从头开始测
            dataloader_reset(&val_loader);
            // 2. 循环测试 val_num_batches 次 (比如 5 次)
            for (int i = 0; i < val_num_batches; i++) {
                dataloader_next_batch(&val_loader);
                // 3. 关键点：只做 Forward，不做 Backward！
                // 因为我们只想知道它错得有多离谱，不想在这里更新权重
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
            }
            // 4. 计算平均损失并打印
            val_loss /= val_num_batches;
            //printf("val loss %f\n", val_loss);
            log_message(log_filename, "step %d val loss %f\n", step, val_loss);
        }

        // once in a while do model inference to print generated text
        // 2. 推理/采样 (Inference) —— “才艺展示”
        if (step > 0 && step % 20 == 0) {   // 每 20 步进行一次
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            // 1. 准备草稿纸：把所有位置填上 <End of Text> 标记，作为起始信号
            for(int i = 0; i < B * T; ++i) {
                gen_tokens[i] = GPT2_EOT;
            }
            // now sample from the model autoregressively
            //printf("generating:\n---\n");
            // === 【修改】记录开始生成 ===
            log_message(log_filename, "generating at step %d:\n---\n", step);
            // 2. 逐字生成循环 (Autoregressive Loop)
            for (int t = 1; t < genT; t++) {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                // A. 前向传播：算出当前的预测概率
                // 注意这里效率极低：每次为了预测第 t 个字，都要把前 t-1 个字重新算一遍
                gpt2_forward(&model, gen_tokens, NULL, B, T);
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // but only using position 0
                // get the V-dimensional vector probs[0, t-1, :]
                // B. 采样：只看第 1 个样本 (b=0) 的第 t-1 个位置的输出
                float* probs = model.acts.probs + (t-1) * model.config.vocab_size;
                float coin = random_f32(&rng_state);    // 掷骰子
                int next_token = sample_mult(probs, model.config.vocab_size, coin); // 选出一个字
                // C. 填入：把新字填进草稿纸
                gen_tokens[t] = next_token;
                // print the generated token, either using the Tokenizer or a fallback
                // D. 打印：即时显示生成的字
                if (tokenizer.init_ok) {
                    const char* token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);

                    // 如果你也想把生成的字写入文件 (可选):
                    FILE* fp = fopen(log_filename, "a");
                    if(fp) { fprintf(fp, "%s", token_str); fclose(fp); }
                } else {
                    // fall back to printing the token id
                    printf("%d ", next_token);
                }
                fflush(stdout);
            }
            //printf("\n---\n");
            log_message(log_filename, "\n---\n"); // 文件里也记录换行
        }

        // --- C. 保存中间检查点 (Checkpoint) ---
        if (save_every > 0 && step > 0 && step % save_every == 0) {
            char ckpt_path[512]; // 申请足够大的缓冲区
            
            // 智能生成路径：基于 output_model_path 修改
            // 假设 output_model_path = "output_model/final_model.bin"
            // 我们希望生成: "output_model/final_model_step_100.bin"
            
            // 1. 查找最后一个 '.' (扩展名) 的位置
            char *dot = strrchr(final_model_path, '.');
            
            if (dot) {
                // 计算 '.' 之前的字符长度
                int base_len = dot - final_model_path;
                // 格式化拼接: "%.*s" (文件名主体) + "_step_%d" (步数) + "%s" (原扩展名)
                snprintf(ckpt_path, sizeof(ckpt_path), "%.*s_step_%d%s", base_len, final_model_path, step, dot);
            } else {
                // 如果文件名没有扩展名 (比如 just "model")，直接追加
                snprintf(ckpt_path, sizeof(ckpt_path), "%s_step_%d", final_model_path, step);
            }
            
            gpt2_save(&model, ckpt_path);
            printf("Saved checkpoint: %s\n", ckpt_path);
        }

        // do a training step
        // 3. 训练步 (Training Step) —— “刷题涨分”
        // 这是最频繁、最核心的一步。只有这一步会改变模型的权重。

        // 1. 计时开始
        auto start = std::chrono::steady_clock::now();
        // 2. 获取数据 (Data)
        dataloader_next_batch(&train_loader);   // 拿到一批新的输入 inputs 和答案 targets
        // 3. 前向传播 (Forward)
        // 算出 logits，并与 targets 比较，算出 loss
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        // 4. 清空旧梯度 (Zero Grad)
        // 非常重要！如果不清空，新的梯度会累加在旧梯度上，导致方向完全错误
        gpt2_zero_grad(&model);
        // 5. 反向传播 (Backward)
        // 从 Loss 开始，沿着网络倒着走，算出每个参数应该怎么改 (grad)
        gpt2_backward(&model);
        // 6. 参数更新 (Update/Optimizer)
        // 实际上这一步是 AdamW 算法：params = params - lr * (m / sqrt(v))
        // step+1 用于 Bias Correction (偏差修正)
        gpt2_update(&model, learning_rate, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
        // 7. 计时结束并打印
        auto end = std::chrono::steady_clock::now();
        double time_elapsed_s = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        //printf("step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
        log_message(log_filename, "step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
    }

    // 保存最终模型
    gpt2_save(&model, final_model_path);

    // free
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    free(gen_tokens);
    return 0;
}
#endif
