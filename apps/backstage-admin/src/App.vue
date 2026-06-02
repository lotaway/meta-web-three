<script setup lang="ts">
import { ElConfigProvider } from 'element-plus'
import { currentElementLocale } from '@/locales'
</script>

<template>
  <!-- Global i18n configuration for Element Plus -->
  <el-config-provider :locale="currentElementLocale">
    <div id="app">
      <router-view />
    </div>
  </el-config-provider>
</template>

<style></style>