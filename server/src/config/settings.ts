import * as path from "node:path";
import os from "node:os";

export default {
    PORT: 30001,
    PROCESS_MAX_COUNT: os.cpus().length / 2,
    PROJECT_DIR: path.join(__dirname, "../../")
}
