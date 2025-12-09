"""
Update Groq models to the latest and best available models.
Based on Groq's current model catalog.
"""
import os
from pathlib import Path

print("=" * 60)
print("Updating to Latest Groq Models")
print("=" * 60)
print()

# Model recommendations based on task
recommendations = {
    "smart": "gpt-oss-120b",  # Most powerful reasoning model
    "fast": "llama-4-scout",  # Fast, versatile, has vision too
    "vision": "llama-4-maverick"  # Best vision model
}

# Check if .env exists
env_file = Path(".env")
if not env_file.exists():
    print("❌ ERROR: .env file not found!")
    exit(1)

print("✅ Found .env file")
print()

# Read existing .env
with open(".env", "r") as f:
    lines = f.readlines()

# Track changes
modified = False
new_lines = []

for line in lines:
    # Update SMART_MODEL
    if line.startswith("SMART_MODEL="):
        old_model = line.split("=")[1].strip()
        new_lines.append(f"SMART_MODEL={recommendations['smart']}\n")
        print(f"✅ SMART_MODEL: {old_model} → {recommendations['smart']}")
        print("   (GPT OSS 120B - Most powerful Groq model)")
        modified = True
    
    # Update FAST_MODEL
    elif line.startswith("FAST_MODEL="):
        old_model = line.split("=")[1].strip()
        new_lines.append(f"FAST_MODEL={recommendations['fast']}\n")
        print(f"✅ FAST_MODEL: {old_model} → {recommendations['fast']}")
        print("   (Llama 4 Scout - Fast & versatile)")
        modified = True
    
    # Update VISION_MODEL (or add if missing)
    elif line.startswith("VISION_MODEL="):
        old_model = line.split("=")[1].strip()
        new_lines.append(f"VISION_MODEL={recommendations['vision']}\n")
        print(f"✅ VISION_MODEL: {old_model} → {recommendations['vision']}")
        print("   (Llama 4 Maverick - Best vision model)")
        modified = True
    
    else:
        new_lines.append(line)

# Add VISION_MODEL if not present
if not any("VISION_MODEL=" in line for line in new_lines):
    # Find where to insert (after other model definitions)
    insert_idx = 0
    for i, line in enumerate(new_lines):
        if "FAST_MODEL=" in line:
            insert_idx = i + 1
            break
    
    if insert_idx > 0:
        new_lines.insert(insert_idx, f"VISION_MODEL={recommendations['vision']}\n")
        print(f"✅ Added VISION_MODEL={recommendations['vision']}")
        print("   (Llama 4 Maverick - Best vision model)")
        modified = True

if modified:
    # Write back
    with open(".env", "w") as f:
        f.writelines(new_lines)
    
    print()
    print("=" * 60)
    print("✅ Successfully updated to latest Groq models!")
    print("=" * 60)
    print()
    print("New configuration:")
    print(f"  🧠 SMART (reasoning): {recommendations['smart']}")
    print(f"  ⚡ FAST (general): {recommendations['fast']}")
    print(f"  👁️  VISION (images): {recommendations['vision']}")
    print()
    print("These are Groq's newest and most powerful models!")
else:
    print()
    print("=" * 60)
    print("ℹ️  Models already up to date or no changes needed")
    print("=" * 60)

print()
print("Benefits of these models:")
print("  • GPT OSS 120B: State-of-the-art reasoning")
print("  • Llama 4 Scout: Fast, multimodal, versatile")
print("  • Llama 4 Maverick: Best vision understanding")
print()
print("Next step:")
print("  🔄 Restart server (Ctrl+C then start.bat)")
print()
