#include "models.h"
#include "llama-memory-recurrent.h"

void llama_model_qwen35moe::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    // Load linear attention (gated delta net) parameters
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // Mark recurrent layers (linear attention layers). MTP layers are dense
    // attention-only and must be flagged non-recurrent.
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    switch (hparams.n_layer()) {
        case 40: type = LLM_TYPE_35B_A3B; break;
        case 48: type = LLM_TYPE_122B_A10B; break;
        case 60: type = LLM_TYPE_397B_A17B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen35moe::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const auto nf = nextn_flags(ml);
    const int trunk_flags = nf.trunk;
    const int mtp_flags   = nf.mtp;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), { n_embd }, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);

    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    auto load_block_trunk = [&](int il, int flags) {
        auto & layer = layers[il];

        const int64_t n_ff_exp   = hparams.n_ff_exp() ? hparams.n_ff_exp() : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        // Calculate dimensions from hyperparameters
        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, flags);

        if (!hparams.is_recr(il)) {
            // Attention layers
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            // Q/K normalization for attention layers
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);
        } else {
            // Linear attention (gated delta net) specific tensors
            // Create tensors with calculated dimensions
            layer.wqkv           = create_tensor(tn(LLM_TENSOR_ATTN_QKV,       "weight", il), { n_embd, key_dim * 2 + value_dim }, TENSOR_NOT_REQUIRED);
            layer.wqkv_gate      = create_tensor(tn(LLM_TENSOR_ATTN_GATE,      "weight", il), { n_embd, value_dim }, TENSOR_NOT_REQUIRED);
            layer.ssm_conv1d     = create_tensor(tn(LLM_TENSOR_SSM_CONV1D,     "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt         = create_tensor(tn(LLM_TENSOR_SSM_DT,         "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a          = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,             il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta       = create_tensor(tn(LLM_TENSOR_SSM_BETA,       "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha      = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,      "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm       = create_tensor(tn(LLM_TENSOR_SSM_NORM,       "weight", il), { head_v_dim }, flags);
            layer.ssm_out        = create_tensor(tn(LLM_TENSOR_SSM_OUT,        "weight", il), { value_dim, n_embd }, flags);
        }

        // Routed experts
        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);

        // Shared experts
        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, flags);
    };

    auto load_block_mtp = [&](int il) {
        auto & layer = layers[il];

        const int64_t n_ff_exp   = hparams.n_ff_exp() ? hparams.n_ff_exp() : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        // MTP block looks like a full-attention Qwen3.5 decoder block with MoE FFN.
        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", il), { n_embd }, mtp_flags);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", il), { n_embd }, mtp_flags);

        create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, mtp_flags);
        layer.wo          = create_tensor(tn(LLM_TENSOR_ATTN_OUT,    "weight", il), { n_embd_head_k * n_head, n_embd }, mtp_flags);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, mtp_flags);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, mtp_flags);

        // Routed experts
        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, mtp_flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, mtp_flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, mtp_flags);

        // Shared experts
        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, mtp_flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, mtp_flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, mtp_flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, mtp_flags);

        // NextN-specific tensors that define the MTP block.
        layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", il), { 2 * n_embd, n_embd }, mtp_flags);
        layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", il), { n_embd },              mtp_flags);
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_vocab },     mtp_flags|TENSOR_NOT_REQUIRED);
        layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), { n_embd },              mtp_flags|TENSOR_NOT_REQUIRED);
    };

    for (int i = 0; i < n_layer; ++i) {
        load_block_trunk(i, trunk_flags);
    }
    for (int i = n_layer; i < n_layer_all; ++i) {
        load_block_mtp(i);
    }

    // Lab switches: projections of the same input stacked row-wise into one weight (no extra memory, the original
    // tensors are views into it), used by graphs of small ubatches (decode, MTP draft and verify), see
    // lab_stk_max_tokens below; only for weights in device buffers. LLAMA_LAB_STACK=1 sets all of them.
    //   LLAMA_LAB_STACK_BA=1     ssm_beta + ssm_alpha         (f32, delta-net layers)
    //   LLAMA_LAB_STACK_QKVZ=1   attn_qkv + attn_gate (z)     (delta-net layers)
    //   LLAMA_LAB_STACK_QKV=1    attn_q + attn_k + attn_v     (attention layers and the MTP block)
    //   LLAMA_LAB_STACK_SHEXP=1  ffn_gate_shexp + ffn_up_shexp (all layers and the MTP block)
    //   LLAMA_LAB_STACK_ROUTER=1 ffn_gate_inp + ffn_gate_inp_shexp (all layers and the MTP block)
    const bool lab_all = llm_lab_flag("LLAMA_LAB_STACK");
    const bool lab_ba  = lab_all || llm_lab_flag("LLAMA_LAB_STACK_BA");
    const bool lab_qz  = lab_all || llm_lab_flag("LLAMA_LAB_STACK_QKVZ");
    const bool lab_qkv = lab_all || llm_lab_flag("LLAMA_LAB_STACK_QKV");
    const bool lab_sh  = lab_all || llm_lab_flag("LLAMA_LAB_STACK_SHEXP");
    const bool lab_rt  = lab_all || llm_lab_flag("LLAMA_LAB_STACK_ROUTER");
    int n_stk = 0;
    for (int il = 0; il < n_layer_all; ++il) {
        auto & layer = layers[il];
        char name[GGML_MAX_NAME];
        auto stk = [&](const char * what) { snprintf(name, sizeof(name), "blk.%d.lab_stk_%s", il, what); return name; };
        if (lab_ba && layer.ssm_beta && layer.ssm_alpha) {
            layer.lab_stk.ba = lab_stack_rows(stk("ba"), { &layer.ssm_beta, &layer.ssm_alpha });
            n_stk += layer.lab_stk.ba != nullptr;
        }
        if (lab_qz && layer.wqkv && layer.wqkv_gate && !layer.wqkv_b) {
            layer.lab_stk.qkvz = lab_stack_rows(stk("qkvz"), { &layer.wqkv, &layer.wqkv_gate });
            n_stk += layer.lab_stk.qkvz != nullptr;
        }
        if (lab_qkv && !layer.wqkv && layer.wq && layer.wk && layer.wv && !layer.wq_b && !layer.wk_b && !layer.wv_b) {
            layer.lab_stk.qkv = lab_stack_rows(stk("qkv"), { &layer.wq, &layer.wk, &layer.wv });
            n_stk += layer.lab_stk.qkv != nullptr;
        }
        if (lab_sh && layer.ffn_gate_shexp && layer.ffn_up_shexp) {
            layer.lab_stk.gu_shexp = lab_stack_rows(stk("gu_shexp"), { &layer.ffn_gate_shexp, &layer.ffn_up_shexp });
            n_stk += layer.lab_stk.gu_shexp != nullptr;
        }
        if (lab_rt && layer.ffn_gate_inp && layer.ffn_gate_inp_shexp) {
            layer.lab_stk.gate_inp = lab_stack_rows(stk("gate_inp"), { &layer.ffn_gate_inp, &layer.ffn_gate_inp_shexp });
            n_stk += layer.lab_stk.gate_inp != nullptr;
        }
    }
    if (lab_ba || lab_qz || lab_qkv || lab_sh || lab_rt) {
        LLAMA_LOG_INFO("%s: lab: %d stacked projection weights\n", __func__, n_stk);
    }
}

// Lab (see load_arch_tensors): the stacked weights are used by graphs of at most this many tokens (decode, MTP
// draft and verify), larger ubatches keep the separate projections (views into the stacks).
static constexpr int64_t lab_stk_max_tokens = 16;

static bool lab_stk_use(const llm_graph_context & g, const ggml_tensor * stk, int64_t max_tokens = lab_stk_max_tokens) {
    return stk != nullptr && g.n_tokens <= max_tokens && (g.loras == nullptr || g.loras->empty());
}

// [ne0*ne1, ne2] -> [ne0, ne1, ne2]: a reshape, or a view if t is a strided view of a stacked projection
static ggml_tensor * lab_split_3d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t ne2) {
    if (ggml_is_contiguous(t)) {
        return ggml_reshape_3d(ctx, t, ne0, ne1, ne2);
    }
    GGML_ASSERT(t->ne[0] == ne0 * ne1 && t->ne[1] == ne2);
    return ggml_view_3d(ctx, t, ne0, ne1, ne2, ggml_row_size(t->type, ne0), t->nb[1], 0);
}

// Q (with the output gate) + K + V of an attention layer; with LLAMA_LAB_STACK_QKV one projection and views
static llm_graph_qkv qwen35moe_build_qkv(llm_graph_context & g, const llama_layer & layer, ggml_tensor * cur,
                                         int64_t n_embd_head, int il) {
    const int64_t n_head    = g.n_head;
    const int64_t n_head_kv = g.n_head_kv;
    if (lab_stk_use(g, layer.lab_stk.qkv) && !layer.wq_s && !layer.wk_s && !layer.wv_s) {
        const int64_t nq = n_embd_head * 2 * n_head;
        const int64_t nk = n_embd_head * n_head_kv;
        ggml_tensor * qkv = g.build_lora_mm(layer.lab_stk.qkv, cur); // [nq + 2*nk, n_tokens]
        g.cb(qkv, "wqkv_stk", il);
        ggml_tensor * q = ggml_view_2d(g.ctx0, qkv, nq, g.n_tokens, qkv->nb[1], 0);
        ggml_tensor * k = ggml_view_2d(g.ctx0, qkv, nk, g.n_tokens, qkv->nb[1], ggml_row_size(qkv->type, nq));
        ggml_tensor * v = ggml_view_2d(g.ctx0, qkv, nk, g.n_tokens, qkv->nb[1], ggml_row_size(qkv->type, nq + nk));
        return { q, k, v };
    }
    return g.build_qkv(layer, cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);
}

// MoE FFN of the trunk and the MTP block: routed experts plus the sigmoid-gated shared expert.
//   LLAMA_LAB_SHEXP_FIRST=1  build the shared expert before the routed experts, so its ADD directly follows the
//                            expert ADDs (and the residual ADD) and joins their MULTI_ADD on Vulkan; same sum order
static ggml_tensor * qwen35moe_build_ffn(llm_graph_context & g, const llama_layer & layer, ggml_tensor * cur, int il, bool mtp) {
    static const bool lab_shexp_first = llm_lab_flag("LLAMA_LAB_SHEXP_FIRST");
    ggml_context * ctx0 = g.ctx0;
    const int64_t n_tokens = cur->ne[1];

    GGML_ASSERT(layer.ffn_gate_inp != nullptr);

    // router + shared expert gate in one projection (single token only: the softmax needs contiguous logits)
    ggml_tensor * logits = nullptr;
    ggml_tensor * shared_gate_raw = nullptr;
    if (lab_stk_use(g, layer.lab_stk.gate_inp, 1) && layer.ffn_gate_inp_b == nullptr) {
        ggml_tensor * r = g.build_lora_mm(layer.lab_stk.gate_inp, cur); // [n_expert + 1, n_tokens]
        g.cb(r, "ffn_gate_inp_stk", il);
        logits          = ggml_view_2d(ctx0, r, g.n_expert, n_tokens, r->nb[1], 0);
        shared_gate_raw = ggml_view_2d(ctx0, r, 1, n_tokens, r->nb[1], ggml_row_size(r->type, g.n_expert));
    }

    auto build_shexp = [&]() -> ggml_tensor * {
        ggml_tensor * sh = nullptr;
        if (lab_stk_use(g, layer.lab_stk.gu_shexp) && !layer.ffn_gate_shexp_s && !layer.ffn_up_shexp_s &&
                !layer.ffn_down_shexp_s && g.hparams.swiglu_clamp_shexp[il] <= 1e-6f) {
            ggml_tensor * gu = g.build_lora_mm(layer.lab_stk.gu_shexp, cur); // [gate; up]
            g.cb(gu, "ffn_gate_up_shexp", il);
            sh = ggml_swiglu(ctx0, gu);
            g.cb(sh, "ffn_swiglu", il);
            sh = g.build_lora_mm(layer.ffn_down_shexp, sh);
        } else {
            sh = g.build_ffn(cur,
                layer.ffn_up_shexp,   NULL, layer.ffn_up_shexp_s,
                layer.ffn_gate_shexp, NULL, layer.ffn_gate_shexp_s,
                layer.ffn_down_shexp, NULL, layer.ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        }
        g.cb(sh, mtp ? "mtp_ffn_shexp" : "ffn_shexp", il);

        // the shared expert has its own gate (one value per token) that is sigmoided
        ggml_tensor * shared_gate = shared_gate_raw ? shared_gate_raw : g.build_lora_mm(layer.ffn_gate_inp_shexp, cur);
        g.cb(shared_gate, "shared_expert_gate", il);

        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        g.cb(shared_gate, mtp ? "mtp_shared_expert_gate_sigmoid" : "shared_expert_gate_sigmoid", il);

        sh = ggml_mul(ctx0, sh, shared_gate);
        g.cb(sh, mtp ? "mtp_ffn_shexp_gated" : "ffn_shexp_gated", il);
        return sh;
    };

    ggml_tensor * ffn_shexp = nullptr;
    if (layer.ffn_up_shexp != nullptr && lab_shexp_first) {
        ffn_shexp = build_shexp();
        ggml_build_forward_expand(g.gf, ffn_shexp);
    }

    ggml_tensor * moe_out =
        g.build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            nullptr,
            g.n_expert, g.n_expert_used,
            LLM_FFN_SILU, true,
            g.hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            logits, layer.ffn_gate_up_exps,
            layer.ffn_up_exps_s,
            layer.ffn_gate_exps_s,
            layer.ffn_down_exps_s);
    g.cb(moe_out, mtp ? "mtp_ffn_moe_out" : "ffn_moe_out", il);

    if (layer.ffn_up_shexp == nullptr) {
        return moe_out;
    }
    if (ffn_shexp == nullptr) {
        ffn_shexp = build_shexp();
    }
    cur = ggml_add(ctx0, moe_out, ffn_shexp);
    g.cb(cur, "ffn_out", il);
    return cur;
}

std::unique_ptr<llm_graph_context> llama_model_qwen35moe::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

llama_model_qwen35moe::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // MTP/NextN layers are loaded as extra decoder blocks but not executed in the main pass.
    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recr(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);

            ggml_build_forward_expand(gf, cur);
        } else {
            ggml_build_forward_expand(gf, cur);

            // Full attention layer
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);
        }

        if (il == n_layer - 1 && crop_before_nextn(inp_out_ids)) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        ggml_tensor * ffn_residual = cur;

        // Post-attention norm
        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // MOE FFN layer
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_moe", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // Input for next layer
        inpL = cur;
    }
    cur = inpL;

    // post-norm hidden state feeds both the LM head and the MTP seed below
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;

    if (crop_after_nextn(inp_out_ids)) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen35moe::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen35moe::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llama_model_qwen35moe::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    auto [Qcur_full, Kcur, Vcur] = qwen35moe_build_qkv(*this, model.layers[il], cur, n_embd_head, il);
    cb(Qcur_full, "Qcur_full", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        Qcur_full->nb[1], 0);
    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    // Apply K normalization
    Kcur = lab_split_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        Qcur_full->nb[1],
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = lab_split_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply IMRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen35moe::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Lab switches for small ubatches (decode, MTP verify) with one sequence in one cell and no state to clear or
    // copy, i.e. when nothing but this layer touches its cache rows (the backend must support the ops):
    //   LLAMA_LAB_CONV_IDS=1     conv state read from the cache by ids and snapshots written back by the conv op
    //                            itself (GGML_OP_SSM_CONV_IDS: no GET_ROWS, CONCAT, separate SILU; the snapshot CPY
    //                            is fused on Vulkan, CONV_CACHE)
    //   LLAMA_LAB_GDN_IDS=1      SSM state read from the cache by ids (ggml_gated_delta_net_ids: no GET_ROWS; the
    //                            snapshot CPY is fused on Vulkan, GDN_CACHE, also without MTP)
    //   LLAMA_LAB_GDN_PROLOGUE=1 (with GDN_IDS) l2 norm of q/k, softplus(alpha + dt) * A and sigmoid(beta) inside
    //                            the GDN op
    static const bool lab_conv_ids = llm_lab_flag("LLAMA_LAB_CONV_IDS");
    static const bool lab_gdn_ids  = llm_lab_flag("LLAMA_LAB_GDN_IDS");
    static const bool lab_gdn_pro  = llm_lab_flag("LLAMA_LAB_GDN_PROLOGUE");
    const int64_t lab_max_tokens = 16;

    const bool rs_inplace = (lab_conv_ids || lab_gdn_ids) && n_seqs == 1 && n_seq_tokens <= lab_max_tokens &&
                            mctx_cur->get_n_rs() == 1 && mctx_cur->get_rs_z() < 0;
    ggml_backend_dev_t layer_dev = rs_inplace ? model.dev_layer(il) : nullptr;

    const auto & layer = model.layers[il];

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    // GDN state read in place (and with it maybe the prologue)?
    const int64_t n_embd_s = hparams.n_embd_s();
    ggml_tensor * ssm_rows = nullptr; // [n_embd_s, n_rows] view of the whole cache when read by ids
    bool gdn_ids = false;
    bool gdn_pro = false;
    bool ba_stk  = false; // beta and alpha from the stacked projection (LLAMA_LAB_STACK_BA)
    if (rs_inplace && lab_gdn_ids && layer_dev != nullptr && ssm_states_all->type == GGML_TYPE_F32) {
        ssm_rows = ggml_reshape_2d(ctx0, ssm_states_all, n_embd_s, ggml_nelements(ssm_states_all) / n_embd_s);
        // probe with stand-ins of the real shapes (the real q/k/v are views of the conv output)
        ggml_tensor * pq = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
        ggml_tensor * pv = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);
        ggml_tensor * pg = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, num_v_heads, n_seq_tokens, n_seqs);
        ggml_tensor * pb = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, num_v_heads, n_seq_tokens, n_seqs);
        ggml_tensor * probe = ggml_gated_delta_net_ids(ctx0, pq, pq, pv, pg, pb, ssm_rows, inp->s_copy_main,
                                                       cparams.n_rs_seq + 1);
        gdn_ids = ggml_backend_dev_supports_op(layer_dev, probe);
        if (gdn_ids && lab_gdn_pro && layer.ssm_dt->type == GGML_TYPE_F32 && layer.ssm_a->type == GGML_TYPE_F32) {
            ggml_gated_delta_net_set_prologue(probe, layer.ssm_dt, layer.ssm_a, 0.0f);
            gdn_pro = ggml_backend_dev_supports_op(layer_dev, probe);
        }
        if (gdn_pro && lab_stk_use(*this, layer.lab_stk.ba) && !layer.ssm_beta_s && !layer.ssm_alpha_s) {
            // the same with beta and alpha as views of the stacked projection (strides, offsets)
            ggml_tensor * sba = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2 * num_v_heads, n_seq_tokens * n_seqs);
            ggml_tensor * vb = ggml_view_4d(ctx0, sba, 1, num_v_heads, n_seq_tokens, n_seqs,
                    sba->nb[0], sba->nb[1], sba->nb[1] * n_seq_tokens, 0);
            ggml_tensor * va = ggml_view_4d(ctx0, sba, 1, num_v_heads, n_seq_tokens, n_seqs,
                    sba->nb[0], sba->nb[1], sba->nb[1] * n_seq_tokens, ggml_row_size(sba->type, num_v_heads));
            ggml_tensor * probe_ba = ggml_gated_delta_net_ids(ctx0, pq, pq, pv, va, vb, ssm_rows, inp->s_copy_main,
                                                              cparams.n_rs_seq + 1);
            ggml_gated_delta_net_set_prologue(probe_ba, layer.ssm_dt, layer.ssm_a, 0.0f);
            ba_stk = ggml_backend_dev_supports_op(layer_dev, probe_ba);
        }
    }

    ggml_tensor * conv_kernel      = layer.ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    // conv state read in place?
    const int64_t conv_row_count = (conv_kernel_size - 1) * conv_channels;
    ggml_tensor * conv_rows = nullptr;
    if (rs_inplace && lab_conv_ids && layer_dev != nullptr && conv_states_all->type == GGML_TYPE_F32 &&
            conv_kernel->type == GGML_TYPE_F32 &&
            conv_states_all->nb[1] == ggml_row_size(GGML_TYPE_F32, conv_row_count)) {
        conv_rows = ggml_reshape_2d(ctx0, conv_states_all, conv_row_count, ggml_nelements(conv_states_all) / conv_row_count);
        // probe with a stand-in of the conv input (the real one may be a strided view of the stacked projection)
        ggml_tensor * px = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, conv_channels, n_seq_tokens, n_seqs);
        ggml_tensor * probe = ggml_ssm_conv_ids(ctx0, px, conv_rows, inp->s_copy_main, conv_kernel,
                                                cparams.n_rs_seq + 1, true);
        if (!ggml_backend_dev_supports_op(layer_dev, probe)) {
            conv_rows = nullptr;
        }
    }

    // Input projections: qkv + z (one stacked projection with LLAMA_LAB_STACK_QKVZ, only with the conv by ids,
    // which reads the strided qkv view directly)
    ggml_tensor * qkv_mixed = nullptr;
    ggml_tensor * z         = nullptr;
    if (conv_rows != nullptr && lab_stk_use(*this, layer.lab_stk.qkvz) && !layer.wqkv_s && !layer.wqkv_gate_s) {
        ggml_tensor * qz = build_lora_mm(layer.lab_stk.qkvz, cur); // [conv_channels + d_inner, n_tokens]
        cb(qz, "linear_attn_qkvz", il);
        const int64_t n_qkv = layer.wqkv->ne[1];
        qkv_mixed = ggml_view_3d(ctx0, qz, n_qkv, n_seq_tokens, n_seqs, qz->nb[1], qz->nb[1] * n_seq_tokens, 0);
        cb(qkv_mixed, "linear_attn_qkv_mixed", il);
        z = ggml_view_2d(ctx0, qz, layer.wqkv_gate->ne[1], n_seq_tokens * n_seqs, qz->nb[1], ggml_row_size(qz->type, n_qkv));
        cb(z, "z", il);
    } else {
        auto qkvz = build_qkvz(cur, il);
        qkv_mixed = qkvz.first;
        z         = qkvz.second;
    }
    GGML_ASSERT(qkv_mixed->type == GGML_TYPE_F32);

    // beta + alpha: one stacked projection with LLAMA_LAB_STACK_BA (with the GDN prologue only, which takes both
    // raw and reads them with the same strides)
    ggml_tensor * beta_raw = nullptr;
    ggml_tensor * alpha    = nullptr;
    if (ba_stk) {
        ggml_tensor * ba = build_lora_mm(layer.lab_stk.ba, cur); // [2*num_v_heads, n_tokens]
        cb(ba, "beta_alpha", il);
        beta_raw = ggml_view_4d(ctx0, ba, 1, num_v_heads, n_seq_tokens, n_seqs,
                ggml_element_size(ba), ba->nb[1], ba->nb[1] * n_seq_tokens, 0);
        alpha    = ggml_view_4d(ctx0, ba, 1, num_v_heads, n_seq_tokens, n_seqs,
                ggml_element_size(ba), ba->nb[1], ba->nb[1] * n_seq_tokens, ggml_row_size(ba->type, num_v_heads));
        cb(beta_raw, "beta", il);
        cb(alpha, "alpha", il);
    } else {
        beta_raw = build_lora_mm(layer.ssm_beta, cur, layer.ssm_beta_s);
        beta_raw = ggml_reshape_4d(ctx0, beta_raw, 1, num_v_heads, n_seq_tokens, n_seqs);
        cb(beta_raw, "beta", il);

        alpha = build_lora_mm(layer.ssm_alpha, cur, layer.ssm_alpha_s);
        alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
        cb(alpha, "alpha", il);
    }

    ggml_tensor * beta = beta_raw;
    ggml_tensor * gate = nullptr;
    if (gdn_pro) {
        // raw, the op does the rest
        gate = alpha->ne[0] == 1 ? alpha : ggml_reshape_4d(ctx0, alpha, 1, num_v_heads, n_seq_tokens, n_seqs);
    } else {
        beta = ggml_sigmoid(ctx0, beta_raw);
        cb(beta, "beta_sigmoid", il);

        ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, layer.ssm_dt);
        ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
        cb(alpha_softplus, "a_softplus", il);

        gate = ggml_mul(ctx0, alpha_softplus, layer.ssm_a);  // -A_log.exp() * softplus
        cb(gate, "gate", il);

        gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);
    }

    ggml_tensor * conv_output_silu = nullptr;
    if (conv_rows != nullptr) {
        // conv output and the K conv state snapshots in one op; the snapshots go to the rollback groups like in
        // build_conv_state (slot j -> group j at kv_head)
        const int64_t K = (int64_t) cparams.n_rs_seq + 1;
        const auto    kv_head  = mctx_cur->get_head();
        const auto    mem_size = mctx_cur->get_size();
        const size_t  row_size = ggml_row_size(GGML_TYPE_F32, conv_row_count);

        ggml_tensor * conv_res = ggml_ssm_conv_ids(ctx0, qkv_mixed, conv_rows, inp->s_copy_main, conv_kernel, K, true);
        cb(conv_res, "conv_ids", il);

        conv_output_silu = ggml_view_3d(ctx0, conv_res, conv_channels, n_seq_tokens, n_seqs,
                ggml_row_size(conv_res->type, conv_channels),
                ggml_row_size(conv_res->type, conv_channels * n_seq_tokens), 0);

        ggml_tensor * snaps = ggml_view_3d(ctx0, conv_res, conv_row_count, n_seqs, K,
                row_size, row_size * n_seqs,
                ggml_row_size(conv_res->type, conv_channels * n_seq_tokens * n_seqs));
        ggml_tensor * dst = ggml_view_3d(ctx0, conv_states_all, conv_row_count, n_seqs, K,
                conv_states_all->nb[1], (size_t) mem_size * row_size, (size_t) kv_head * row_size);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, snaps, dst));
    } else {
        ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

        ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
        cb(conv_output_proper, "conv_output_raw", il);

        conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    }
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * state = nullptr;
    if (gdn_ids) {
        state = ssm_rows;
    } else {
        state = build_rs(inp, ssm_states_all, n_embd_s, n_seqs, nullptr, /*allow_inplace=*/true);
        state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
        cb(state, "state_predelta", il);
    }

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);


    const float eps_norm = hparams.f_norm_rms_eps;

    if (!gdn_pro) {
        q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
        k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);
    }

    //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    // note: need explicit repeat only if we are not using the fused GDN.
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = gdn_ids ?
        build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il, inp->s_copy_main,
                gdn_pro ? model.layers[il].ssm_dt : nullptr, gdn_pro ? model.layers[il].ssm_a : nullptr,
                eps_norm / (float) head_k_dim) :
        build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_is_contiguous(z) ?
        ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs) :
        ggml_view_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
                ggml_row_size(z->type, head_v_dim), z->nb[1], z->nb[1] * n_seq_tokens, 0); // view of the stacked qkvz

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // 2-D: with a [d, n_seq_tokens, n_seqs] src1 the projection is n_seqs matmuls that each read the whole ssm_out weight
    ggml_tensor * final_output = ggml_reshape_2d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens * n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    // Reshape back to original dimensions
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen35moe::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    return qwen35moe_build_ffn(*this, model.layers[il], cur, il, false);
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for Qwen3.5/3.6 MoE
llama_model_qwen35moe::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN35MOE MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN35MOE MTP currently only supports a single MTP block");

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj    && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm      && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm      && "MTP block missing nextn.hnorm");
    GGML_ASSERT(layer.ffn_gate_inp     && "MTP block missing ffn_gate_inp");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    // TODO: extract in a common llm_graph_context::build_inp_embd_h()
    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_inp(), hparams.n_embd);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);

    // TODO: make static using `ggml_build_forward_select()`
    //       see llm_graph_context::build_inp_embd() for reference
    ggml_tensor * tok_embd;
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.token) {
        ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;

        tok_embd = build_get_rows(tok_embd_w, inp->tokens);
    } else {
        tok_embd = inp->embd;
    }
    cb(tok_embd, "mtp_tok_embd", il);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * h_embd = inp->h;

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * h_norm = build_norm(h_embd, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    cb(h_norm, "mtp_hnorm", il);

    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    cb(e_norm, "mtp_enorm", il);

    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    ggml_tensor * cur = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    cb(cur, "mtp_eh_proj", il);

    ggml_tensor * inpSA = cur;

    cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_norm", il);

    auto [Qcur_full, Kcur, Vcur] = qwen35moe_build_qkv(*this, layer, cur, n_embd_head, il);
    cb(Qcur_full, "mtp_Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            Qcur_full->nb[1],
            0);
    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "mtp_Qcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full,
            n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            Qcur_full->nb[1],
            ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "mtp_gate", il);

    Kcur = lab_split_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "mtp_Kcur_normed", il);

    Vcur = lab_split_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
    cb(Vcur, "mtp_Vcur", il);

    Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    const float kq_scale = hparams.f_attention_scale == 0.0f
            ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp_attn,
            nullptr, nullptr, nullptr,
            Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "mtp_attn_pregate", il);

    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
    cur = build_lora_mm(layer.wo, cur, layer.wo_s);
    cb(cur, "mtp_attn_out", il);

    cur = ggml_add(ctx0, cur, inpSA);
    cb(cur, "mtp_attn_residual", il);

    ggml_tensor * ffn_residual = cur;
    cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "mtp_attn_post_norm", il);

    // MoE FFN — routed experts plus gated shared expert (mirrors qwen35moe).
    cur = qwen35moe_build_ffn(*this, layer, cur, il, true);
    cb(cur, "mtp_ffn_out", il);

    cur = ggml_add(ctx0, cur, ffn_residual);
    cb(cur, "mtp_post_ffn", il);

    ggml_tensor * head_norm_w = layer.nextn.shared_head_norm
            ? layer.nextn.shared_head_norm
            : model.output_norm;
    GGML_ASSERT(head_norm_w && "QWEN35MOE MTP: missing both nextn.shared_head_norm and output_norm");
    cur = build_norm(cur, head_norm_w, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "h_nextn", -1);
    res->t_h_nextn= cur;

    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    cb(cur, "mtp_shared_head_norm", -1);

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN35MOE MTP: missing LM head (nextn.shared_head_head or model.output)");
    if (mtp_draft != nullptr && head_w == model.output && n_outputs == 1) {
        // this context drafts over a row subset of the LM head: scatter its logits into a full-vocab row of -inf
        GGML_ASSERT(mtp_draft->n_keep == cparams.mtp_draft_vocab);
        const int64_t n_sel = mtp_draft->head->ne[1];
        const int64_t n_row = model.output->ne[1];
        ggml_tensor * sub = build_lora_mm(mtp_draft->head, cur, head_s);
        cur = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_row), -INFINITY);
        cur = ggml_set_rows(ctx0, cur, ggml_reshape_2d(ctx0, sub, 1, n_sel), mtp_draft->ids);
        cur = ggml_reshape_2d(ctx0, cur, n_row, 1);
    } else {
        cur = build_lora_mm(head_w, cur, head_s);
    }
    cb(cur, "result_output", -1);

    res->t_logits = cur;
}
