<script setup lang="ts">
import { ref, watch } from 'vue'
import { ElMessage, type UploadProps, type UploadUserFile } from 'element-plus'
import { MESSAGE_DURATION_SHORT } from '@/constants'
import { t } from '@/locales'

const props = defineProps({
  modelValue: {
    type: String,
    default: ''
  }
})

const emit = defineEmits(['update:modelValue'])
const dialogVisible = ref(false)
const uploadUrl = import.meta.env.VITE_BASE_SERVER_URL + import.meta.env.VITE_UPLOAD_URL
const fileList = ref<UploadUserFile[]>([])

watch(() => props.modelValue, (newVal) => {
  if (newVal) {
    const fileName = newVal.substring(newVal.lastIndexOf("/") + 1)
    fileList.value = [{
      name: fileName,
      url: newVal
    }]
  } else {
    fileList.value = []
  }
}, { immediate: true })

const emitInput = (val: string) => {
  emit('update:modelValue', val)
}

const handleRemove: UploadProps['onRemove'] = () => {
  emitInput('')
}

const handlePreview: UploadProps['onPreview'] = () => {
  dialogVisible.value = true
}

const handleUploadSuccess: UploadProps['onSuccess'] = (res, file) => {
  if (res.code === 500) {
    ElMessage({
      message: t('upload.uploadFailed'),
      type: 'error',
      duration: MESSAGE_DURATION_SHORT,
    })
    return
  }
  fileList.value.pop()
  fileList.value.push({ name: file.name, url: res.data })
  emitInput(res.data)
}
</script>

<template>
  <div>
    <el-upload :action="uploadUrl" list-type="picture" :multiple="false"
      :show-file-list="props.modelValue ? true : false" :file-list="fileList"
      :on-remove="handleRemove" :on-success="handleUploadSuccess" :on-preview="handlePreview">
      <el-button size="small" type="primary">{{ t('upload.clickToUpload') }}</el-button>
      <template #tip>
        <div class="el-upload__tip">{{ t('upload.tip') }}</div>
      </template>
    </el-upload>
    <el-dialog v-model="dialogVisible">
      <img width="100%" :src="fileList[0]?.url" :alt="fileList[0]?.name || 'preview'">
    </el-dialog>
  </div>
</template>