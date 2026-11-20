<script lang="ts" setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import SidebarItem from './SidebarItem.vue'
import ScrollBar from '@/components/ScrollBar/index.vue'
import { useAppStore } from '@/stores/app'
import usePermissionStore from '@/stores/permission'

// Define component name
defineOptions({
  name: 'Sidebar'
})

// Use Pinia stores
const appStore = useAppStore()
const permissionStore = usePermissionStore()

const route = useRoute()
// All routes
const routes = computed(() => permissionStore.routers)
// Sidebar open state
const isCollapse = computed(() => !appStore.sidebar.opened)
</script>

<template>
  <scroll-bar>
    <el-menu mode="vertical" :show-timeout="200" :default-active="route.path" :collapse="isCollapse"
      background-color="#304156" text-color="#bfcbd9" active-text-color="#409EFF">
      <sidebar-item :routes="routes"></sidebar-item>
    </el-menu>
  </scroll-bar>
</template>
