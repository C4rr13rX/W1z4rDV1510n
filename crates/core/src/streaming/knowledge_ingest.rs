use crate::network::compute_payload_hash;
use crate::schema::Timestamp;
use crate::streaming::knowledge::{FigureAsset, KnowledgeDocument, TextBlock};
use anyhow::{Context, Result};
use quick_xml::events::Event;
use quick_xml::Reader;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct KnowledgeIngestConfig {
    pub source: String,
    pub asset_root: Option<PathBuf>,
    pub require_image_bytes: bool,
    pub normalize_whitespace: bool,
    pub include_ocr_blocks: bool,
    pub ocr_command: Option<Vec<String>>,
    pub ocr_timeout_secs: u64,
}

impl Default for KnowledgeIngestConfig {
    fn default() -> Self {
        Self {
            source: "NLM".to_string(),
            asset_root: None,
            require_image_bytes: true,
            normalize_whitespace: false,
            include_ocr_blocks: false,
            ocr_command: None,
            ocr_timeout_secs: 30,
        }
    }
}

pub struct NlmJatsIngestor {
    config: KnowledgeIngestConfig,
}

impl NlmJatsIngestor {
    pub fn new(config: KnowledgeIngestConfig) -> Self {
        Self { config }
    }

    pub fn parse_str(&self, xml: &str, timestamp: Timestamp) -> Result<KnowledgeDocument> {
        let bytes = xml.as_bytes();
        self.parse_bytes(bytes, timestamp)
    }

    pub fn parse_bytes(&self, xml: &[u8], timestamp: Timestamp) -> Result<KnowledgeDocument> {
        let mut reader = Reader::from_reader(xml);
        reader.trim_text(false);
        let mut buf = Vec::new();

        let mut metadata: HashMap<String, Value> = HashMap::new();
        let mut doc_title = String::new();
        let mut article_id_type: Option<String> = None;
        let mut article_id_value = String::new();
        let mut in_article_title = false;
        let mut in_sec_title = false;
        let mut in_paragraph = false;
        let mut in_caption = false;
        let mut in_fig_label = false;
        let mut current_section: Vec<String> = Vec::new();
        let mut sec_title_buf = String::new();
        let mut paragraph_buf = String::new();
        let mut paragraph_refs: Vec<String> = Vec::new();
        let mut caption_buf = String::new();
        let mut fig_label_buf = String::new();
        let mut figures: Vec<FigureAsset> = Vec::new();
        let mut text_blocks: Vec<TextBlock> = Vec::new();
        let mut block_order: usize = 0;
        let mut fig_order: usize = 0;

        let mut fig_builder: Option<FigureBuilder> = None;

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    let tag = tag_name(&reader, e.name());
                    match tag.as_str() {
                        "article-id" => {
                            article_id_type = attr_value(&reader, e, "pub-id-type");
                            article_id_value.clear();
                        }
                        "article-title" => {
                            in_article_title = true;
                        }
                        "sec" => {
                            current_section.push(String::new());
                        }
                        "title" => {
                            if !current_section.is_empty() && fig_builder.is_none() {
                                in_sec_title = true;
                                sec_title_buf.clear();
                            }
                        }
                        "p" => {
                            if in_caption {
                                // Caption text handled separately.
                            } else {
                                in_paragraph = true;
                                paragraph_buf.clear();
                                paragraph_refs.clear();
                            }
                        }
                        "xref" => {
                            if in_paragraph {
                                let ref_type = attr_value(&reader, e, "ref-type")
                                    .unwrap_or_default()
                                    .to_ascii_lowercase();
                                if ref_type == "fig" || ref_type == "figure" {
                                    if let Some(rid) = attr_value(&reader, e, "rid") {
                                        for token in rid.split_whitespace() {
                                            if !token.trim().is_empty() {
                                                paragraph_refs.push(token.trim().to_string());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        "fig" => {
                            let fig_id = attr_value(&reader, e, "id").unwrap_or_default();
                            fig_builder = Some(FigureBuilder::new(fig_id));
                            fig_label_buf.clear();
                            caption_buf.clear();
                        }
                        "label" => {
                            if fig_builder.is_some() {
                                in_fig_label = true;
                                fig_label_buf.clear();
                            }
                        }
                        "caption" => {
                            if fig_builder.is_some() {
                                in_caption = true;
                                caption_buf.clear();
                            }
                        }
                        "graphic" | "inline-graphic" | "media" => {
                            if let Some(fig) = fig_builder.as_mut() {
                                if fig.image_ref.is_none() {
                                    fig.image_ref = attr_value(&reader, e, "xlink:href")
                                        .or_else(|| attr_value(&reader, e, "href"));
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Ok(Event::Empty(ref e)) => {
                    let tag = tag_name(&reader, e.name());
                    match tag.as_str() {
                        "xref" => {
                            if in_paragraph {
                                let ref_type = attr_value(&reader, e, "ref-type")
                                    .unwrap_or_default()
                                    .to_ascii_lowercase();
                                if ref_type == "fig" || ref_type == "figure" {
                                    if let Some(rid) = attr_value(&reader, e, "rid") {
                                        for token in rid.split_whitespace() {
                                            if !token.trim().is_empty() {
                                                paragraph_refs.push(token.trim().to_string());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        "graphic" | "inline-graphic" | "media" => {
                            if let Some(fig) = fig_builder.as_mut() {
                                if fig.image_ref.is_none() {
                                    fig.image_ref = attr_value(&reader, e, "xlink:href")
                                        .or_else(|| attr_value(&reader, e, "href"));
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Ok(Event::Text(e)) => {
                    let text = e.unescape().unwrap_or_default().to_string();
                    if in_article_title {
                        doc_title.push_str(&text);
                    } else if in_sec_title {
                        sec_title_buf.push_str(&text);
                    } else if in_paragraph {
                        paragraph_buf.push_str(&text);
                    } else if in_caption {
                        caption_buf.push_str(&text);
                    } else if in_fig_label {
                        fig_label_buf.push_str(&text);
                    } else if article_id_type.is_some() {
                        article_id_value.push_str(&text);
                    }
                }
                Ok(Event::End(ref e)) => {
                    let tag = tag_name(&reader, e.name());
                    match tag.as_str() {
                        "article-id" => {
                            if let Some(id_type) = article_id_type.take() {
                                if !article_id_value.trim().is_empty() {
                                    metadata.insert(
                                        id_type.to_ascii_lowercase(),
                                        Value::String(article_id_value.trim().to_string()),
                                    );
                                }
                            }
                        }
                        "article-title" => {
                            in_article_title = false;
                        }
                        "title" => {
                            if in_sec_title {
                                in_sec_title = false;
                                if let Some(last) = current_section.last_mut() {
                                    *last = normalize_text(&sec_title_buf, self.config.normalize_whitespace);
                                }
                            }
                        }
                        "p" => {
                            if in_paragraph {
                                in_paragraph = false;
                                let text = normalize_text(&paragraph_buf, self.config.normalize_whitespace);
                                if !text.trim().is_empty() {
                                    paragraph_refs.sort();
                                    paragraph_refs.dedup();
                                    let section = current_section_path(&current_section);
                                    let block_id = compute_payload_hash(
                                        format!("block|{}|{}", block_order, text).as_bytes(),
                                    );
                                    text_blocks.push(TextBlock {
                                        block_id,
                                        text,
                                        section,
                                        order: block_order,
                                        figure_refs: paragraph_refs.clone(),
                                        source: "xml".to_string(),
                                        confidence: 1.0,
                                    });
                                    block_order += 1;
                                }
                            }
                        }
                        "sec" => {
                            current_section.pop();
                        }
                        "label" => {
                            if in_fig_label {
                                in_fig_label = false;
                                if let Some(fig) = fig_builder.as_mut() {
                                    if !fig_label_buf.trim().is_empty() {
                                        fig.label = Some(normalize_text(
                                            &fig_label_buf,
                                            self.config.normalize_whitespace,
                                        ));
                                    }
                                }
                            }
                        }
                        "caption" => {
                            if in_caption {
                                in_caption = false;
                                if let Some(fig) = fig_builder.as_mut() {
                                    if !caption_buf.trim().is_empty() {
                                        fig.caption = Some(normalize_text(
                                            &caption_buf,
                                            self.config.normalize_whitespace,
                                        ));
                                    }
                                }
                            }
                        }
                        "fig" => {
                            if let Some(fig) = fig_builder.take() {
                                if let Some((figure, ocr_text)) =
                                    finalize_figure(fig, fig_order, &self.config)
                                {
                                    figures.push(figure.clone());
                                    fig_order += 1;
                                    if self.config.include_ocr_blocks {
                                        if let Some(text) = ocr_text {
                                            let block_id = compute_payload_hash(
                                                format!("ocr|{}|{}", figure.figure_id, text).as_bytes(),
                                            );
                                            text_blocks.push(TextBlock {
                                                block_id,
                                                text,
                                                section: Some("figure_ocr".to_string()),
                                                order: block_order,
                                                figure_refs: vec![figure.figure_id.clone()],
                                                source: "ocr".to_string(),
                                                confidence: 0.5,
                                            });
                                            block_order += 1;
                                        }
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Ok(Event::Eof) => break,
                Err(err) => return Err(err).context("failed to parse JATS XML"),
                _ => {}
            }
            buf.clear();
        }

        if !doc_title.trim().is_empty() {
            metadata.insert("title".to_string(), Value::String(doc_title.trim().to_string()));
        }
        metadata.insert("source".to_string(), Value::String(self.config.source.clone()));
        metadata.insert("ingested_at".to_string(), Value::from(timestamp.unix));

        let doc_id = resolve_doc_id(&metadata, xml);
        Ok(KnowledgeDocument {
            doc_id,
            source: self.config.source.clone(),
            title: if doc_title.trim().is_empty() {
                None
            } else {
                Some(doc_title.trim().to_string())
            },
            text_blocks,
            figures,
            metadata,
        })
    }
}

impl Default for NlmJatsIngestor {
    fn default() -> Self {
        Self::new(KnowledgeIngestConfig::default())
    }
}

#[derive(Debug, Clone)]
struct FigureBuilder {
    id: String,
    label: Option<String>,
    caption: Option<String>,
    image_ref: Option<String>,
}

impl FigureBuilder {
    fn new(id: String) -> Self {
        Self {
            id,
            label: None,
            caption: None,
            image_ref: None,
        }
    }
}

fn finalize_figure(
    fig: FigureBuilder,
    order: usize,
    config: &KnowledgeIngestConfig,
) -> Option<(FigureAsset, Option<String>)> {
    let image_ref = fig.image_ref?;
    let image_bytes = resolve_image_bytes(&image_ref, config.asset_root.as_deref());
    if config.require_image_bytes && image_bytes.is_none() {
        return None;
    }
    let image_hash = if let Some(bytes) = &image_bytes {
        compute_payload_hash(bytes)
    } else {
        compute_payload_hash(format!("ref|{image_ref}").as_bytes())
    };
    let ocr_text = image_bytes
        .as_ref()
        .and_then(|bytes| run_ocr(config, &image_ref, bytes).ok().flatten());
    let figure = FigureAsset {
        figure_id: if fig.id.trim().is_empty() {
            fig.label.clone().unwrap_or_else(|| format!("fig-{order}"))
        } else {
            fig.id.clone()
        },
        label: fig.label,
        caption: fig.caption,
        image_ref,
        image_hash,
        order,
        ocr_text: ocr_text.clone(),
    };
    Some((figure, ocr_text))
}

fn resolve_image_bytes(image_ref: &str, asset_root: Option<&Path>) -> Option<Vec<u8>> {
    if image_ref.contains("://") {
        return None;
    }
    let path = if let Some(root) = asset_root {
        root.join(image_ref)
    } else {
        PathBuf::from(image_ref)
    };
    if !path.exists() {
        return None;
    }
    fs::read(&path).ok()
}

fn run_ocr(config: &KnowledgeIngestConfig, image_ref: &str, bytes: &[u8]) -> Result<Option<String>> {
    let Some(command) = &config.ocr_command else {
        return Ok(None);
    };
    if command.is_empty() {
        return Ok(None);
    }
    let image_path = materialize_image(image_ref, bytes)?;
    let (bin, args) = resolve_command(command, &image_path);
    let output = std::process::Command::new(bin)
        .args(args)
        .output()
        .context("failed to run OCR command")?;
    if !output.status.success() {
        return Ok(None);
    }
    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if text.is_empty() {
        Ok(None)
    } else {
        Ok(Some(text))
    }
}

fn materialize_image(image_ref: &str, bytes: &[u8]) -> Result<PathBuf> {
    let mut path = PathBuf::from(image_ref);
    if path.exists() {
        return Ok(path);
    }
    let hash = compute_payload_hash(bytes);
    path = std::env::temp_dir().join(format!("ocr_{hash}.bin"));
    if !path.exists() {
        fs::write(&path, bytes).context("failed to write OCR temp file")?;
    }
    Ok(path)
}

fn resolve_command(command: &[String], image_path: &Path) -> (String, Vec<String>) {
    let mut args = Vec::new();
    let image_str = image_path.to_string_lossy();
    let bin = command.first().cloned().unwrap_or_default();
    for arg in command.iter().skip(1) {
        args.push(arg.replace("{image}", &image_str));
    }
    (bin, args)
}

fn resolve_doc_id(metadata: &HashMap<String, Value>, xml: &[u8]) -> String {
    for key in ["pmid", "pmc", "doi", "publisher-id"] {
        if let Some(val) = metadata.get(key).and_then(|v| v.as_str()) {
            if !val.trim().is_empty() {
                return format!("{key}:{}", val.trim());
            }
        }
    }
    compute_payload_hash(xml)
}

fn current_section_path(section_stack: &[String]) -> Option<String> {
    let parts: Vec<String> = section_stack
        .iter()
        .filter(|val| !val.trim().is_empty())
        .cloned()
        .collect();
    if parts.is_empty() {
        None
    } else {
        let joined = parts.join(" > ");
        Some(joined)
    }
}

fn normalize_text(text: &str, normalize_whitespace: bool) -> String {
    let trimmed = text.trim();
    if !normalize_whitespace {
        return trimmed.to_string();
    }
    let mut out = String::new();
    let mut last_space = false;
    for ch in trimmed.chars() {
        if ch.is_whitespace() {
            if !last_space {
                out.push(' ');
                last_space = true;
            }
        } else {
            out.push(ch);
            last_space = false;
        }
    }
    out
}

fn tag_name(reader: &Reader<&[u8]>, name: quick_xml::name::QName) -> String {
    reader.decoder().decode(name.as_ref()).unwrap_or_default().to_string()
}

fn attr_value(reader: &Reader<&[u8]>, e: &quick_xml::events::BytesStart, key: &str) -> Option<String> {
    for attr in e.attributes().flatten() {
        let attr_key = reader.decoder().decode(attr.key.as_ref()).ok()?;
        if attr_key == key || attr_key.ends_with(key) {
            let value = attr.unescape_value().ok()?;
            return Some(value.to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_jats_with_figure_refs() {
        let xml = r#"
        <article>
          <front>
            <article-meta>
              <article-id pub-id-type="pmid">12345</article-id>
              <title-group>
                <article-title>Sample Title</article-title>
              </title-group>
            </article-meta>
          </front>
          <body>
            <sec>
              <title>Results</title>
              <p>See <xref ref-type="fig" rid="F1">Fig. 1</xref> for details.</p>
              <fig id="F1">
                <label>Figure 1</label>
                <caption><p>Caption text.</p></caption>
                <graphic xlink:href="fig1.png" />
              </fig>
            </sec>
          </body>
        </article>
        "#;
        let mut config = KnowledgeIngestConfig::default();
        config.require_image_bytes = false;
        let ingestor = NlmJatsIngestor::new(config);
        let doc = ingestor.parse_str(xml, Timestamp { unix: 1 }).expect("doc");
        assert!(doc.doc_id.starts_with("pmid:"));
        assert_eq!(doc.figures.len(), 1);
        assert_eq!(doc.text_blocks.len(), 1);
        assert_eq!(doc.text_blocks[0].figure_refs, vec!["F1"]);
    }
}
