from semantixel.api import create_app
from semantixel.core.config import config

app = create_app()

if __name__ == "__main__":
    # Local development run
    app.run(host="0.0.0.0", port=config.port)
