<script lang="ts" setup>
import { useI18n } from 'vue-i18n'
const { t } = useI18n()
import { useRouter } from 'vue-router'
import img_404 from '@/assets/images/gif_404.gif'

// Define component name
defineOptions({
  name: 'wrongPage'
})

// Use router
const router = useRouter()

// Method definitions
const handleGoMain = () => {
  router.push({ path: '/' })
}
</script>
<template>
  <div>
    <div class="app-container">
      <el-row>
        <el-col :span="12">
          <img :src="img_404" alt="404" class="img-style">
        </el-col>
        <el-col :span="12">
          <div style="margin-left: 100px;margin-top: 60px">
            <h1 class="color-main">OOPS!</h1>
            <h2 style="color: #606266">{{ t('common.notFoundTitle') }}</h2>
            <div style="color:#909399;font-size: 14px">{{ t('common.notFoundDesc') }}</div>
            <el-button style="margin-top: 20px" type="primary" round @click="handleGoMain">{{ t('common.backHome') }}</el-button>
          </div>
        </el-col>
      </el-row>
    </div>
  </div>
</template>

<style lang="scss" scoped>
.app-container {
  width: 80%;
  margin: 120px auto;
}

.img-style {
  width: auto;
  height: auto;
  max-width: 100%;
  max-height: 100%;
}
</style>
