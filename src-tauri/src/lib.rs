use serde::{Deserialize, Serialize};
use std::process::{Command, Child};
use std::sync::Mutex;
use tauri::Manager;

// State to hold the child process so it can be killed on exit
struct AppState {
    pub bridge_process: Mutex<Option<Child>>,
}

#[derive(Serialize, Deserialize)]
struct AnalysisResult {
    omni_count: usize,
    wds_count: usize,
    total_count: usize,
    output_path: String,
    mask_path: Option<String>,
}

#[derive(Serialize)]
struct AnalyzePayload {
    #[serde(rename = "imagePath")]
    image_path: String,
}

#[tauri::command]
async fn load_models() -> Result<String, String> {
    let client = reqwest::Client::new();
    let res = client.post("http://127.0.0.1:8991/load_models")
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if res.status().is_success() {
        Ok("loaded".to_string())
    } else {
        Err(format!("Error loading models: {}", res.status()))
    }
}

#[tauri::command]
async fn paste_clipboard_image() -> Result<String, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .map_err(|e| e.to_string())?;
        
    let res = client.post("http://127.0.0.1:8991/paste_clipboard")
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if res.status().is_success() {
        let json: serde_json::Value = res.json().await.map_err(|e| e.to_string())?;
        if let Some(path) = json.get("path").and_then(|v| v.as_str()) {
            Ok(path.to_string())
        } else {
            Err("No path returned from clipboard paste".into())
        }
    } else {
        Err(format!("Error pasting clipboard: {}", res.status()))
    }
}

#[tauri::command]
async fn run_analysis(image_path: String) -> Result<AnalysisResult, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300)) // Model inference takes time
        .build()
        .map_err(|e| e.to_string())?;

    let payload = AnalyzePayload { image_path };
    
    let res = client.post("http://127.0.0.1:8991/analyze")
        .json(&payload)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if res.status().is_success() {
        let result: AnalysisResult = res.json().await.map_err(|e| e.to_string())?;
        Ok(result)
    } else {
        let err_text = res.text().await.unwrap_or_default();
        Err(format!("Analysis failed: {}", err_text))
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            // Start the Python server
            let manifest_dir = std::env::current_dir().unwrap_or_default();
            let py_script = manifest_dir.join("../OmniParser/tauri_bridge.py");
            
            // Try to spawn the sidecar using `python`
            let child = Command::new("python")
                .arg(&py_script)
                .spawn()
                .ok(); // Ignore if it fails, maybe it's running already or python isn't in PATH

            app.manage(AppState {
                bridge_process: Mutex::new(child),
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            load_models,
            paste_clipboard_image,
            run_analysis
        ])
        .on_window_event(|app, event| match event {
            tauri::WindowEvent::Destroyed => {
                // Kill process on exit
                if let Ok(mut child_lock) = app.state::<AppState>().bridge_process.lock() {
                    if let Some(mut child) = child_lock.take() {
                        let _ = child.kill();
                    }
                }
            }
            _ => {}
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
