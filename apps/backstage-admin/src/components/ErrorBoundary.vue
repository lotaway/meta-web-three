<script setup lang="ts">
import { onErrorCaptured, ref } from 'vue'
import { normalizeError } from '@/utils/errorBoundary'

const hasError = ref(false)
const detail = ref('')

onErrorCaptured((error) => {
  hasError.value = true
  detail.value = normalizeError(error).message
  console.error('Unhandled render error:', error)
  return false
})

function reload(): void {
  window.location.reload()
}
</script>

<template>
  <div v-if="hasError" class="error-boundary" role="alert">
    <h2>Something went wrong</h2>
    <p>The page hit an unexpected error and could not be displayed.</p>
    <p v-if="detail" class="error-boundary__detail">{{ detail }}</p>
    <button type="button" class="error-boundary__reload" aria-label="Reload page" @click="reload">
      Reload page
    </button>
  </div>
  <slot v-else />
</template>

<style scoped>
.error-boundary {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  min-height: 60vh;
  padding: 24px;
  text-align: center;
}

.error-boundary__detail {
  opacity: 0.7;
  word-break: break-word;
}

.error-boundary__reload {
  padding: 8px 20px;
  cursor: pointer;
}
</style>
