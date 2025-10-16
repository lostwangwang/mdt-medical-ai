<!-- frontend/src/components/InputBox.vue -->
<template>
  <div class="input-box">
    <div class="input-container">
      <textarea
        ref="inputElement"
        v-model="inputText"
        :placeholder="placeholder"
        :disabled="disabled"
        @keydown.enter.exact.prevent="handleSend"
        @keydown.enter.shift.exact="addNewLine"
        @keydown="handleKeyDown"
        @focus="handleFocus"
        @blur="handleBlur"
        rows="1"
        aria-label="输入您的问题"
      ></textarea>

      <button
        class="send-button"
        :disabled="!inputText.trim() || disabled"
        @click="handleSend"
        aria-label="发送消息"
        :aria-disabled="!inputText.trim() || disabled"
      >
        <svg
          v-if="!disabled"
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
        >
          <path
            d="M22 2L11 13"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
          <path
            d="M22 2L15 22L11 13L2 9L22 2Z"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
        </svg>
        <span v-else class="loading-spinner"></span>
      </button>
    </div>

    <div class="input-hints">
      <span class="hint">按 Enter 发送，Shift + Enter 换行，Ctrl + Enter 发送</span>
      <span class="char-count" :class="{ warning: inputText.length > 900 }">
        {{ inputText.length }} / 1000
      </span>
    </div>
  </div>
</template>

<script>
import { ref, watch, onMounted, onUnmounted } from "vue";

export default {
  name: "InputBox",
  props: {
    placeholder: {
      type: String,
      default: "请输入您的问题...",
    },
    disabled: {
      type: Boolean,
      default: false,
    },
  },
  emits: ["send-message"],
  setup(props, { emit }) {
    const inputText = ref("");
    const inputElement = ref(null);
    const resizeTimeout = ref(null);
    
    // 自动调整textarea高度的函数
    const resizeTextarea = () => {
      if (!inputElement.value) return;
      
      // 清除之前的定时器
      if (resizeTimeout.value) {
        clearTimeout(resizeTimeout.value);
      }
      
      // 使用防抖优化
      resizeTimeout.value = setTimeout(() => {
        inputElement.value.style.height = "auto";
        // 限制最大高度为150px
        const maxHeight = 150;
        const scrollHeight = inputElement.value.scrollHeight;
        inputElement.value.style.height = Math.min(scrollHeight, maxHeight) + "px";
      }, 50);
    };

    const handleSend = () => {
      if (!inputText.value.trim() || props.disabled) return;

      emit("send-message", inputText.value.trim());
      inputText.value = "";

      // 重置textarea高度
      if (inputElement.value) {
        inputElement.value.style.height = "auto";
      }
    };

    const addNewLine = () => {
      inputText.value += "\n";
      // 触发resize以更新高度
      resizeTextarea();
    };

    // 监听输入变化，自动调整高度
    watch(inputText, () => {
      resizeTextarea();
    });

    // 焦点处理
    const handleFocus = () => {
      inputElement.value?.parentElement?.classList.add('focused');
    };

    const handleBlur = () => {
      inputElement.value?.parentElement?.classList.remove('focused');
    };

    // 键盘快捷键支持
    const handleKeyDown = (event) => {
      // Ctrl/Cmd + Enter 发送消息
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        handleSend();
        return;
      }
      
      // 其他处理由模板中的修饰符处理
    };

    onMounted(() => {
      // 组件挂载时初始化resize
      resizeTextarea();
    });

    onUnmounted(() => {
      // 清理定时器
      if (resizeTimeout.value) {
        clearTimeout(resizeTimeout.value);
      }
    });

    return {
      inputText,
      inputElement,
      handleSend,
      addNewLine,
      handleFocus,
      handleBlur,
      handleKeyDown
    };
  },
};
</script>

<style scoped>
.input-box {
  padding: 1rem 1.5rem;
  border-top: 1px solid #e9ecef;
  background: #f8f9fa;
}

.input-container {
  display: flex;
  gap: 0.75rem;
  align-items: flex-end;
  background: white;
  border-radius: 24px;
  padding: 0.75rem 1rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  border: 2px solid transparent;
  transition: border-color 0.3s, box-shadow 0.3s;
}

.input-container:focus-within,
.input-container.focused {
  border-color: #667eea;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

textarea {
  flex: 1;
  border: none;
  outline: none;
  resize: none;
  font-size: 1rem;
  font-family: inherit;
  max-height: 150px;
  overflow-y: auto;
  line-height: 1.5;
  transition: color 0.3s;
}

textarea:disabled {
  background: transparent;
  color: #999;
}

textarea::placeholder {
  color: #adb5bd;
  transition: color 0.3s;
}

textarea:focus::placeholder {
  color: #cbd5e0;
}

textarea::-webkit-scrollbar {
  width: 4px;
}

textarea::-webkit-scrollbar-track {
  background: transparent;
}

textarea::-webkit-scrollbar-thumb {
  background: #ccc;
  border-radius: 2px;
}

textarea::-webkit-scrollbar-thumb:hover {
  background: #999;
}

.send-button {
  flex-shrink: 0;
  width: 40px;
  height: 40px;
  border: none;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border-radius: 50%;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: transform 0.2s, box-shadow 0.2s, background 0.3s;
  position: relative;
  overflow: hidden;
}

.send-button:not(:disabled):hover {
  transform: scale(1.05);
  box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.send-button:not(:disabled):active {
  transform: scale(0.98);
}

.send-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
  transform: none;
}

.send-button svg {
  transition: transform 0.3s;
}

.send-button:not(:disabled):hover svg {
  transform: scale(1.1);
}

.loading-spinner {
  width: 20px;
  height: 20px;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: white;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.input-hints {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 0.5rem;
  padding: 0 0.5rem;
  font-size: 0.75rem;
  color: #6c757d;
  transition: color 0.3s;
}

.char-count {
  transition: color 0.3s, font-weight 0.3s;
}

.char-count.warning {
  color: #f5576c;
  font-weight: 600;
  animation: pulse-warning 1s infinite;
}

@keyframes pulse-warning {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: 0.7;
  }
}

/* 移动端适配 */
@media (max-width: 768px) {
  .input-box {
    padding: 0.75rem 1rem;
  }
  
  .input-container {
    padding: 0.5rem 0.75rem;
    gap: 0.5rem;
  }
  
  .send-button {
    width: 36px;
    height: 36px;
  }
  
  .input-hints {
    flex-direction: column;
    gap: 0.25rem;
    align-items: flex-start;
    font-size: 0.7rem;
  }
}
</style>
