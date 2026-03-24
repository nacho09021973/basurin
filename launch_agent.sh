#!/bin/bash
# launch_agent.sh — Lanza Claude Code en tmux para trabajo autónomo local
#
# Uso:
#   ./launch_agent.sh                    # Lanza nueva sesión
#   ./launch_agent.sh attach             # Reconecta a sesión existente
#   ./launch_agent.sh status             # Muestra si hay sesión activa
#
# Después de lanzar:
#   - Se abre tmux con el venv activado en ~/work/basurin
#   - Escribe: claude
#   - Dale la tarea: "Lee CLAUDE.md y README_diario_221.md.
#     Ejecuta la siguiente tarea pendiente de prioridad alta."
#   - Desconecta con Ctrl+B, D
#   - Reconecta cuando quieras: ./launch_agent.sh attach

SESSION_NAME="basurin-agent"
WORKDIR="$HOME/work/basurin"

case "${1:-launch}" in
    attach)
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            tmux attach -t "$SESSION_NAME"
        else
            echo "No hay sesión activa '$SESSION_NAME'."
            echo "Usa: ./launch_agent.sh  para crear una nueva."
            exit 1
        fi
        ;;
    status)
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            echo "✓ Sesión '$SESSION_NAME' activa."
            tmux list-windows -t "$SESSION_NAME"
        else
            echo "✗ No hay sesión '$SESSION_NAME'."
        fi
        ;;
    launch|"")
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            echo "Ya existe sesión '$SESSION_NAME'. Reconectando..."
            tmux attach -t "$SESSION_NAME"
            exit 0
        fi

        if [ ! -d "$WORKDIR" ]; then
            echo "ERROR: No existe $WORKDIR"
            exit 1
        fi

        echo "Lanzando sesión '$SESSION_NAME' en $WORKDIR..."
        tmux new-session -d -s "$SESSION_NAME" -c "$WORKDIR"
        tmux send-keys -t "$SESSION_NAME" "source .venv/bin/activate" Enter
        tmux send-keys -t "$SESSION_NAME" "echo '=== BASURIN agent ready ==='" Enter
        tmux send-keys -t "$SESSION_NAME" "echo 'Escribe: claude'" Enter
        tmux attach -t "$SESSION_NAME"
        ;;
    *)
        echo "Uso: $0 [launch|attach|status]"
        exit 1
        ;;
esac
