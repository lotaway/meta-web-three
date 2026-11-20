<script setup lang="ts">
import { ElConfigProvider } from 'element-plus'
import { currentElementLocale } from '@/locales'
import ErrorBoundary from '@/components/ErrorBoundary.vue'
</script>

<template>
  <!-- Global i18n configuration for Element Plus -->
  <el-config-provider :locale="currentElementLocale">
    <ErrorBoundary>
      <div id="app">
        <router-view />
      </div>
    </ErrorBoundary>
  </el-config-provider>
</template>

<style></style>