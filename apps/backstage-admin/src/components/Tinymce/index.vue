<template>
  <div class="rich-editor">
    <editor v-model="contentValue" :init="initOptions" :disabled="disabled" license-key="gpl" @onClick="onClick"
      @onBlur="onBlur" @onFocus="onFocus" />
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import Editor from '@tinymce/tinymce-vue'

// Bundle TinyMCE and its required components through Vite instead of loading a
// prebuilt script from the public directory.
import 'tinymce'
import 'tinymce/icons/default'
import 'tinymce/themes/silver'
import 'tinymce/models/dom'

// Register the bundled skins as TinyMCE resources so no runtime CSS request is made.
import 'tinymce/skins/ui/oxide/skin.js'
import 'tinymce/skins/ui/oxide/content.js'
import 'tinymce/skins/content/default/content.js'

// Plugins enabled by default in this editor.
import 'tinymce/plugins/link'
import 'tinymce/plugins/image'
import 'tinymce/plugins/table'
import 'tinymce/plugins/lists'
import 'tinymce/plugins/code'
import 'tinymce/plugins/fullscreen'

// Simplified Chinese translations, registered into the global TinyMCE instance.
import 'tinymce-i18n/langs7/zh_CN.js'

import { createImageUploadHandler } from './upload'
import { BASE_SERVER_URL, UPLOAD_URL } from '@/utils/env'

const uploadUrl = BASE_SERVER_URL + UPLOAD_URL

interface Props {
  modelValue: string
  disabled?: boolean
  height?: number
  toolbar?: string[]
  plugins?: string[]
  placeholder?: string
}

interface Emits {
  (e: 'update:modelValue', value: string): void
  (e: 'click'): void
  (e: 'blur'): void
  (e: 'focus'): void
}

const props = withDefaults(defineProps<Props>(), {
  disabled: false,
  height: 400,
  toolbar: () => [
    'blocks | undo redo remove| bold italic underline strikethrough removeformat | fontselect fontsizeselect formatselect | alignleft aligncenter alignright alignjustify',
    'fontsize forecolor backcolor bullist numlist |link image table | code fullscreen'
  ],
  plugins: () => [
    'link', 'image', 'table', 'lists', 'code', 'fullscreen'
  ],
  placeholder: ''
})

const emit = defineEmits<Emits>()

const contentValue = ref(props.modelValue)

// Sync external value changes into the editor.
watch(
  () => props.modelValue,
  (newValue) => {
    if (newValue !== contentValue.value) {
      contentValue.value = newValue
    }
  }
)

// Sync editor changes back to the parent component.
watch(contentValue, (newValue) => {
  emit('update:modelValue', newValue)
})

const initOptions = {
  language: 'zh_CN',
  height: props.height,
  menubar: false,
  plugins: props.plugins,
  toolbar: props.toolbar,
  placeholder: props.placeholder,
  branding: false, // Hide the TinyMCE brand logo.
  resize: true, // Allow manual resizing.
  elementpath: false, // Hide the element path in the status bar.
  content_style: `
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
      font-size: 14px;
      margin: 10px;
      line-height: 1.6;
    }
  `,
  images_upload_handler: createImageUploadHandler(uploadUrl)
}

const onClick = () => {
  emit('click')
}

const onBlur = () => {
  emit('blur')
}

const onFocus = () => {
  emit('focus')
}
</script>

<style scoped>
.rich-editor {
  border: 1px solid #dcdfe6;
  border-radius: 4px;
}

:deep(.tox-tinymce) {
  border-radius: 4px;
  border: none;
}

:deep(.tox-edit-area) {
  min-height: 300px;
}
</style>
