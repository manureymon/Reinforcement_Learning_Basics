import gymnasium as gym
from BJ import BlackjackEnv  # Importa tu entorno desde BJ.py

# Crear el entorno con renderizado activado
env = BlackjackEnv(render_mode="human")

# Reiniciar el entorno
obs, _ = env.reset()

# Jugar una ronda automática
done = False
while not done:
    action = env.action_space.sample()  # Elegir acción aleatoria (0=stick, 1=hit)
    obs, reward, done, _, _ = env.step(action)

env.close()  # Cierra la ventana al terminar