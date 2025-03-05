rm -rf .venv
curl -LsSf https://astral.sh/uv/install.sh | sh
make regen-lock
make install
