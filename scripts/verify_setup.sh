#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}[Aeon] Verifying setup...${NC}"

# Check dependencies
if ! command -v docker &> /dev/null; then
    echo -e "${RED}[Error] Docker is not installed.${NC}"
    exit 1
fi

echo -e "${GREEN}[Aeon] Building Docker image...${NC}"
# Use the runtime stage for verification so we know the final artifact works
docker build -f deploy/Dockerfile -t aeon-verify:latest .

echo -e "${GREEN}[Aeon] Running verification container...${NC}"
# Explicitly run the command to print info
OUTPUT=$(docker run --rm aeon-verify:latest python -c "import aeon_py.core; info = aeon_py.core.get_build_info(); print(info)")

if [[ $? -eq 0 && ! -z "$OUTPUT" ]]; then
    echo -e "${GREEN}[Success] Verification Passed!${NC}"
    echo "Output from Core:"
    echo "$OUTPUT"
else
    echo -e "${RED}[Failure] Verification Failed.${NC}"
    exit 1
fi

# Cleanup
echo -e "${GREEN}[Aeon] Cleaning up...${NC}"
docker rmi aeon-verify:latest > /dev/null 2>&1

exit 0
