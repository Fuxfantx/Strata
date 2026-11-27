// Exercise the importer against the real Hy4 pack: it should accept the backbone file, report the family's architecture,
// and treat the companion as a companion rather than a missing shard.
import { inspectLocalModel } from '../../apps/host/server/local-models.mjs';

const pack = 'C:/Users/Fuxfantx/C-Res/Qwen/Hy4-Flash-Compact/Hy4-flash-Uncensored-Compact-00001-of-00002.gguf';

try {
    const info = inspectLocalModel(pack);
    const local = info.variant?.local ?? info.local;
    console.log('accepted:', local.architecture, '| files', local.files.length,
                '| ctx', local.contextLength, '| shards', local.shards,
                '| warnings', info.warnings?.length ?? 0);
} catch (e) {
    console.log('rejected:', e.message);
}
