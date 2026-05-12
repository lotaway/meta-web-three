<script setup lang="ts">
import { ref, watch } from 'vue'
import { Plus } from '@element-plus/icons-vue'
import { ElMessage, type UploadProps, type UploadUserFile } from 'element-plus'
import { MESSAGE_DURATION_SHORT } from '@/constants'
import { t } from '@/locales'

const props = defineProps({
  modelValue: {
    type: Array<string>,
    default: []
  },
  maxCount: {
    type: Number,
    default: 5
  }
})

const emit = defineEmits(['update:modelValue'])
const dialogVisible = ref(false)
const dialogImageUrl = ref('')
const uploadUrl = import.meta.env.VITE_BASE_SERVER_URL + import.meta.env.VITE_UPLOAD_URL
const fileList = ref<UploadUserFile[]>([])

watch(() => props.modelValue, (newVal) => {
  if (newVal) {
    fileList.value = newVal.map(item => {
      const fileName = item.substring(item.lastIndexOf("/") + 1)
      return { name: fileName, url: item }
    })
  } else {
    fileList.value = []
  }
}, { immediate: true })

const emitInput = (val: string[]) => {
  emit('update:modelValue', val)
}

const handleRemove: UploadProps['onRemove'] = (file, fileList) => {
  const remainingFiles = fileList.filter(item => item.uid !== file.uid)
  fileList = remainingFiles
  const urls = remainingFiles.map(item => item.url || '')
  emitInput(urls)
}

const handlePreview: UploadProps['onPreview'] = (file) => {
  dialogVisible.value = true
  dialogImageUrl.value = file.url!
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
  fileList.value.push({ name: file.name, url: res.data })
  const urls = fileList.value.map(item => item.url || '')
  emitInput(urls)
}
</script>

<template>
  <div>
    <el-upload :action="uploadUrl" list-type="picture-card" :file-list="fileList"
      :limit="maxCount" :on-remove="handleRemove" :on-success="handleUploadSuccess"
      :on-preview="handlePreview">
      <el-icon>
        <Plus />
      </el-icon>
    </el-upload>
    <el-dialog v-model="dialogVisible">
      <img :src="dialogImageUrl" :alt="dialogImageUrl || 'preview'">
    </el-dialog>
  </div>
</template>
