git config --global user.email "pvikram035@gmail.com"
git config --global user.name "Vikram Pawar"
export TERM=xterm-256color
rm -rf .venv
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -LsSf https://hf.co/cli/install.sh | bash
source $HOME/.local/bin/env
make regen-lock
make install
make dev