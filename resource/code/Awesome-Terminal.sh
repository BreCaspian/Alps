# command custom —— 时间 + 状态 emoji

if [ -z "${ORIG_PS1_SAVED+x}" ]; then
  ORIG_PS1_SAVED="$PS1"
fi

prompt_status() {
  local last=$?   # 上一条命令的退出码
  case "$last" in
    0)   PS1_STATUS="😀" ;;          # 成功
    1)   PS1_STATUS="⚠️ " ;;         # 一般错误
    2)   PS1_STATUS="⛔" ;;          # 命令用法错误
    126) PS1_STATUS="🔐" ;;          # 权限不够
    127) PS1_STATUS="❓" ;;          # 命令不存在
    130) PS1_STATUS="⏹  " ;;         # Ctrl+C 中断
    137) PS1_STATUS="💀" ;;          # kill -9 / OOM
    139) PS1_STATUS="💥" ;;          # 段错误
    *)   PS1_STATUS="❌ ${last}" ;;  # 其他错误
  esac
}

case ";$PROMPT_COMMAND;" in
  *";prompt_status;"*)
    ;;
  *)
    if [ -n "$PROMPT_COMMAND" ]; then
      PROMPT_COMMAND="prompt_status; $PROMPT_COMMAND"
    else
      PROMPT_COMMAND="prompt_status"
    fi
    ;;
esac

PS1='\D{%H:%M} ${PS1_STATUS} '"$ORIG_PS1_SAVED"
