"""
Quick script to disable LangSmith tracing to prevent timeout errors.
"""
import os
from pathlib import Path

print("=" * 60)
print("Disable LangSmith Tracing (Fix Timeout Errors)")
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

# Track changes
modified = False
new_lines = []
has_disable_tracing = False

for line in lines:
    if line.startswith("DISABLE_TRACING="):
        new_lines.append("DISABLE_TRACING=1\n")
        print("✅ Set DISABLE_TRACING=1")
        modified = True
        has_disable_tracing = True
    elif line.startswith("LANGSMITH_OTEL_ENABLED="):
        new_lines.append("LANGSMITH_OTEL_ENABLED=false\n")
        print("✅ Disabled LANGSMITH_OTEL_ENABLED")
        modified = True
    elif line.startswith("LANGSMITH_TRACING="):
        new_lines.append("LANGSMITH_TRACING=false\n")
        print("✅ Disabled LANGSMITH_TRACING")
        modified = True
    else:
        new_lines.append(line)

# Add DISABLE_TRACING if not present
if not has_disable_tracing:
    new_lines.append("\n# Disable LangSmith tracing to prevent timeout errors\n")
    new_lines.append("DISABLE_TRACING=1\n")
    print("✅ Added DISABLE_TRACING=1")
    modified = True

if modified:
    # Write back
    with open(".env", "w") as f:
        f.writelines(new_lines)
    
    print()
    print("=" * 60)
    print("✅ LangSmith tracing disabled!")
    print("=" * 60)
    print()
    print("This will prevent timeout errors from api.smith.langchain.com")
else:
    print()
    print("=" * 60)
    print("ℹ️  Tracing already disabled")
    print("=" * 60)

print()
print("Next step:")
print("  🔄 Restart server for changes to take effect")
print()
