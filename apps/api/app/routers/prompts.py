from fastapi import APIRouter, HTTPException
from ..core.db import query
import json
from typing import Optional
from datetime import datetime
import difflib  # NEW

router = APIRouter(prefix="/prompts", tags=["prompts"])

@router.get("")
def list_prompts(name: Optional[str] = None):
    """
    List all prompts or filter by name.
    Returns prompts ordered by name ASC, version DESC.
    """
    if name:
        rows = query("""
            SELECT name, version, system, user_template, params, created_at
            FROM prompts
            WHERE name = %s
            ORDER BY version DESC;
        """, (name,))
    else:
        rows = query("""
            SELECT name, version, system, user_template, params, created_at
            FROM prompts
            ORDER BY name ASC, version DESC;
        """)
    
    return {"prompts": [{
        "name": r[0],
        "version": r[1],
        "system": r[2],
        "user_template": r[3],
        "params": r[4] or {},
        "created_at": r[5].isoformat() if r[5] else None
    } for r in rows]}

@router.get("/{name}/{version}")
def get_prompt(name: str, version: int):
    """
    Get a specific prompt version.
    """
    rows = query("""
        SELECT name, version, system, user_template, params, created_at
        FROM prompts
        WHERE name = %s AND version = %s;
    """, (name, version))
    
    if not rows:
        raise HTTPException(404, detail=f"Prompt {name}@{version} not found")
    
    r = rows[0]
    return {
        "name": r[0],
        "version": r[1],
        "system": r[2],
        "user_template": r[3],
        "params": r[4] or {},
        "created_at": r[5].isoformat() if r[5] else None
    }

@router.post("")
def create_prompt_version(body: dict):
    """
    Create a new prompt version.
    Input:
      {
        "name": "rag",
        "source_version": 1,  // optional: copy from this version
        "system": "...",      // optional: override system
        "user_template": "...", // optional: override template
        "params": {...}       // optional: override params
      }
    Output: {"name": "rag", "version": 2}
    """
    name = body.get("name")
    if not name:
        raise HTTPException(400, detail="name required")
    
    source_version = body.get("source_version")
    system = body.get("system")
    user_template = body.get("user_template")
    params = body.get("params")
    
    # Determine next version
    rows = query("SELECT MAX(version) FROM prompts WHERE name=%s;", (name,))
    max_ver = rows[0][0] if rows and rows[0][0] else 0
    new_version = max_ver + 1
    
    # If source_version specified, copy from it
    if source_version is not None:
        src_rows = query("""
            SELECT system, user_template, params
            FROM prompts
            WHERE name=%s AND version=%s;
        """, (name, source_version))
        if not src_rows:
            raise HTTPException(404, detail=f"Source prompt {name}@{source_version} not found")
        src_system, src_template, src_params = src_rows[0]
        # Override with provided values
        if system is None:
            system = src_system
        if user_template is None:
            user_template = src_template
        if params is None:
            params = src_params
    
    # Validate required fields
    if system is None:
        raise HTTPException(400, detail="system required (or source_version to copy from)")
    if user_template is None:
        raise HTTPException(400, detail="user_template required (or source_version to copy from)")
    
    # Insert new version
    try:
        query("""
            INSERT INTO prompts(name, version, system, user_template, params)
            VALUES (%s, %s, %s, %s, %s);
        """, (name, new_version, system, user_template, json.dumps(params or {})))
    except Exception as ex:
        # Handle unique constraint violation
        if "unique" in str(ex).lower():
            raise HTTPException(409, detail=f"Prompt {name}@{new_version} already exists")
        raise HTTPException(500, detail=f"Failed to create prompt: {ex}")
    
    return {"name": name, "version": new_version}

@router.delete("/{name}/{version}")
def delete_prompt_version(name: str, version: int):
    """
    Delete a specific prompt version (soft delete - don't actually remove if referenced).
    For now: hard delete if no runs reference it.
    """
    # Check if prompt exists
    rows = query("SELECT version FROM prompts WHERE name=%s AND version=%s;", (name, version))
    if not rows:
        raise HTTPException(404, detail=f"Prompt {name}@{version} not found")
    
    # Check if any runs reference it
    run_rows = query("""
        SELECT COUNT(*) FROM runs
        WHERE config::jsonb @> %s::jsonb;
    """, (json.dumps({"prompt": {"name": name, "version": version}}),))
    
    if run_rows and run_rows[0][0] > 0:
        raise HTTPException(400, detail=f"Prompt {name}@{version} is referenced by {run_rows[0][0]} run(s) and cannot be deleted")
    
    query("DELETE FROM prompts WHERE name=%s AND version=%s;", (name, version))
    return {"deleted": True, "name": name, "version": version}

@router.get("/diff")  # NEW
def diff_prompts(name: str, a: int, b: int):
    rows = query("""
        SELECT version, system, user_template, params
        FROM prompts
        WHERE name=%s AND version IN (%s,%s)
        ORDER BY version ASC;
    """ % ("%s", "%s", "%s"), (name, a, b))
    if len(rows) != 2:
        raise HTTPException(404, detail="Both versions required")
    v1, sys1, usr1, params1 = rows[0]
    v2, sys2, usr2, params2 = rows[1]
    def _lines(x): return (x or "").splitlines()
    system_diff = list(difflib.unified_diff(_lines(sys1), _lines(sys2), fromfile=f"{name}@{v1}:system", tofile=f"{name}@{v2}:system", lineterm=""))
    user_diff = list(difflib.unified_diff(_lines(usr1), _lines(usr2), fromfile=f"{name}@{v1}:user", tofile=f"{name}@{v2}:user", lineterm=""))
    params1 = params1 or {}
    params2 = params2 or {}
    added = {k: params2[k] for k in params2.keys() - params1.keys()}
    removed = {k: params1[k] for k in params1.keys() - params2.keys()}
    changed = {k: (params1[k], params2[k]) for k in params1.keys() & params2.keys() if params1[k] != params2[k]}
    return {
        "name": name,
        "version_a": v1,
        "version_b": v2,
        "system_diff": system_diff,
        "user_template_diff": user_diff,
        "params_added": added,
        "params_removed": removed,
        "params_changed": changed
    }
