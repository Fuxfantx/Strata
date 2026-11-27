// Print what the importer's reader returns for the keys it validates, so a rejection can be read rather than guessed.
import { readGGUF } from '../../apps/host/server/gguf.mjs';

const pack = 'C:/Users/Fuxfantx/C-Res/Qwen/Hy4-Flash-Compact/Hy4-flash-Uncensored-Compact-00001-of-00002.gguf';
const r = readGGUF(pack);
for (const k of ['general.architecture', 'hy_v4_flash.attention.head_count', 'hy_v4_flash.context_length',
                 'hy_v4_flash.block_count', 'hy_v4_flash.embedding_length', 'hy_v4_flash.expert_count',
                 'hy_v4_flash.attention.head_count_kv', 'hy_v4_flash.attention.key_length']) {
    const v = r.metadata[k];
    console.log(k, '->', Array.isArray(v) ? `array(${v.length}) first ${v.slice(0, 4)}` : JSON.stringify(v) ?? 'undefined');
}
