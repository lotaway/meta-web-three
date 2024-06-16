use farmfe_plugin::{Plugin, PluginContext, PluginConfig, PluginHookResult, PluginLoadArgs};
use std::path::Path;

pub struct WasmPlugin;

impl Plugin for WasmPlugin {
    fn name(&self) -> &'static str {
        "wasm-plugin"
    }

    fn load(&self, args: PluginLoadArgs, ctx: &PluginContext) -> PluginHookResult {
        let file_path = Path::new(&args.path);
        if file_path.extension().and_then(|s| s.to_str()) == Some("wasm") {
            let wasm_content = std::fs::read(file_path).expect("Failed to read WASM file");
            return PluginHookResult::Handled {
                contents: Some(wasm_content),
                content_type: Some("application/wasm".to_string()),
            };
        }
        PluginHookResult::Pass
    }
}

#[no_mangle]
pub fn create_plugin(_config: PluginConfig) -> Box<dyn Plugin> {
    Box::new(WasmPlugin)
}
