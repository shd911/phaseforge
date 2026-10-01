//! b141.71 (audit stage 2 A7): every field the frontend writes into a project
//! must survive Rust's ProjectFile round trip (serde drops unknown fields
//! silently — b141.18 lost the delay-convention marker that way). The input
//! is produced by src/lib/__tests__/project-contract.test.ts.

use phaseforge_lib::project::ProjectFile;
use serde_json::Value;

/// Drop nulls so "absent" and "null" compare equal (serde skips None).
fn strip_nulls(v: &Value) -> Value {
    match v {
        Value::Object(m) => Value::Object(m.iter().filter(|(_, x)| !x.is_null())
            .map(|(k, x)| (k.clone(), strip_nulls(x))).collect()),
        Value::Array(a) => Value::Array(a.iter().map(strip_nulls).collect()),
        x => x.clone(),
    }
}

fn missing(path: &str, a: &Value, b: &Value, out: &mut Vec<String>) {
    match (a, b) {
        (Value::Object(ma), Value::Object(mb)) => {
            for (k, va) in ma {
                match mb.get(k) {
                    None => out.push(format!("{path}.{k}")),
                    Some(vb) => missing(&format!("{path}.{k}"), va, vb, out),
                }
            }
        }
        (Value::Array(xa), Value::Array(xb)) => {
            for (i, (va, vb)) in xa.iter().zip(xb).enumerate() { missing(&format!("{path}[{i}]"), va, vb, out); }
        }
        (Value::Number(x), Value::Number(y)) => {
            if (x.as_f64().unwrap_or(0.0) - y.as_f64().unwrap_or(0.0)).abs() > 1e-12 { out.push(format!("{path} ({x} → {y})")); }
        }
        (x, y) if x != y => out.push(format!("{path} ({x} → {y})")),
        _ => {}
    }
}

#[test]
fn every_frontend_field_survives_the_rust_round_trip() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ts_project_contract.json");
    let Ok(src) = std::fs::read_to_string(path) else {
        eprintln!("[project_contract] fixture missing — run `npx vitest run project-contract` first");
        return;
    };
    let original: Value = serde_json::from_str(&src).expect("json");
    let project: ProjectFile = serde_json::from_value(original.clone()).expect("ProjectFile parse");
    let back = serde_json::to_value(&project).expect("serialize");
    let mut lost = Vec::new();
    missing("$", &strip_nulls(&original), &strip_nulls(&back), &mut lost);
    assert!(lost.is_empty(), "fields lost or changed in the Rust round trip:\n  {}", lost.join("\n  "));
}
