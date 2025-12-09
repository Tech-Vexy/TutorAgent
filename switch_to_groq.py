"""
Quick script to switch from Google Gemini to Groq models.
Run this to avoid quota errors.
"""
import os
from pathlib import Path

print("=" * 60)
print("Switch to Groq Models (Fix Quota Error)")
print("=" * 60)
print()

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

# Track if we made changes
modified = False
new_lines = []

for line in lines:
    # Replace MODEL_PROVIDER
    if line.startswith("MODEL_PROVIDER="):
        if "google" in line.lower():
            new_lines.append("MODEL_PROVIDER=groq\n")
            print("✅ Changed MODEL_PROVIDER from google to groq")
            modified = True
        else:
            new_lines.append(line)
    
    # Update SMART_MODEL if it's a Gemini model
    elif line.startswith("SMART_MODEL="):
        if "gemini" in line.lower():
            new_lines.append("SMART_MODEL=llama-3.3-70b-versatile\n")
            print("✅ Changed SMART_MODEL to llama-3.3-70b-versatile")
            modified = True
        else:
            new_lines.append(line)
    
    # Update FAST_MODEL if it's a Gemini model
    elif line.startswith("FAST_MODEL="):
        if "gemini" in line.lower():
            new_lines.append("FAST_MODEL=llama-3.1-8b-instant\n")
            print("✅ Changed FAST_MODEL to llama-3.1-8b-instant")
            modified = True
        else:
            new_lines.append(line)
    
    else:
        new_lines.append(line)

# Check if Groq models are set
has_smart = any("SMART_MODEL=" in line for line in new_lines)
has_fast = any("FAST_MODEL=" in line for line in new_lines)
has_provider = any("MODEL_PROVIDER=" in line for line in new_lines)

if not has_provider:
    new_lines.append("\n# Model Provider\n")
    new_lines.append("MODEL_PROVIDER=groq\n")
    print("✅ Added MODEL_PROVIDER=groq")
    modified = True

if not has_smart:
    new_lines.append("SMART_MODEL=llama-3.3-70b-versatile\n")
    print("✅ Added SMART_MODEL=llama-3.3-70b-versatile")
    modified = True

if not has_fast:
    new_lines.append("FAST_MODEL=llama-3.1-8b-instant\n")
    print("✅ Added FAST_MODEL=llama-3.1-8b-instant")
    modified = True

if modified:
    # Write back
    with open(".env", "w") as f:
        f.writelines(new_lines)
    print()
    print("=" * 60)
    print("✅ Successfully switched to Groq!")
    print("=" * 60)
else:
    print()
    print("=" * 60)
    print("ℹ️  Already using Groq or no changes needed")
    print("=" * 60)

print()
print("Next steps:")
print("1. RESTART your server (Ctrl+C then start.bat)")
print("2. Look for: 'Using Groq Models'")
print("3. Test - quota error should be gone!")
print()
print("Groq models are:")
print("  • FASTER than Gemini (seriously!)")
print("  • FREE with 14,400 requests/day")
print("  • No quota issues")
print()
