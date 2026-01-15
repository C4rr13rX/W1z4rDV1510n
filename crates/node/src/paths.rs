use std::path::PathBuf;

pub fn node_data_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("W1Z4RDV1510N_DATA_DIR") {
        return PathBuf::from(dir);
    }
    if cfg!(target_os = "windows") {
        if let Some(appdata) = std::env::var_os("APPDATA") {
            return PathBuf::from(appdata).join("w1z4rdv1510n");
        }
    } else if cfg!(target_os = "macos") {
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join("w1z4rdv1510n");
        }
    } else {
        if let Some(xdg) = std::env::var_os("XDG_DATA_HOME") {
            return PathBuf::from(xdg).join("w1z4rdv1510n");
        }
        if let Some(home) = std::env::var_os("HOME") {
            return PathBuf::from(home)
                .join(".local")
                .join("share")
                .join("w1z4rdv1510n");
        }
    }
    PathBuf::from("w1z4rdv1510n-data")
}
