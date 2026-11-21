<script lang="ts" setup>
import type { RouteRecordExt } from '@/types/router'
import { computed, type PropType } from 'vue'
import { useI18n } from 'vue-i18n'

// Define component name
defineOptions({
  name: 'SidebarItem'
})

const { t } = useI18n()

// Translate title: translate if it is a key, otherwise display as-is
const translateTitle = (title: string | undefined) => {
  if (!title) return ''
  // If it is in translation-key format (contains a dot), translate it
  if (title.includes('.') && !title.includes(' ')) {
    return t(title)
  }
  return title
}

// Safe path handling
const toStr = (v: unknown): string => {
  if (v === null || v === undefined) return ''
  if (typeof v === 'object') return ''
  return String(v)
}

// Build path
const joinPath = (...parts: unknown[]): string => {
  return parts.map(p => toStr(p)).filter(Boolean).join('/')
}

// Define props
const props = defineProps({
  // Generate menu routes
  routes: {
    type: Array as PropType<RouteRecordExt[]>
  },
  // Control the style of top-level menus with a single child
  isNest: {
    type: Boolean,
    default: false
  }
})

// Preprocess route data into string paths
interface ProcessedRoute {
  path: string
  name: string
  children?: ProcessedRoute[]
  meta?: { title?: string; icon?: string }
  hidden?: boolean
  alwaysShow?: boolean
}

const processedRoutes = computed((): ProcessedRoute[] => {
  const process = (route: RouteRecordExt): ProcessedRoute => ({
    path: toStr(route.path),
    name: toStr(route.name),
    children: route.children?.map(process),
    meta: route.meta as ProcessedRoute['meta'],
    hidden: route.hidden,
    alwaysShow: route.alwaysShow
  })
  
  return props.routes!
    .filter(item => !item.hidden && item.children)
    .map(process)
})

// Check whether a route has only one child route
const hasOneShowingChildren = (children: ProcessedRoute[]) => {
  const showingChildren = children.filter(item => !item.hidden)
  return showingChildren.length === 1
}

// Get filtered child routes
const getFilteredChildren = (children: ProcessedRoute[]) => {
  return children.filter(child => !child.hidden)
}

// Compute menu index
const calcIndex = (item: ProcessedRoute): string => {
  return item.name || item.path
}

// Compute child route index
const calcChildIndex = (parent: ProcessedRoute, child: ProcessedRoute): string => {
  return joinPath(parent.path, child.path)
}
</script>

<template>
  <div class="menu-wrapper">
    <template v-for="item in processedRoutes">
      <!-- Top-level menu with a single child -->
      <router-link
        v-if="item.children && hasOneShowingChildren(item.children) && !item.children[0]?.children && !item.alwaysShow"
        :to="calcChildIndex(item, item.children[0]!)" :key="item.children[0]?.name">
        <el-menu-item :index="calcChildIndex(item, item.children[0]!)"
          :class="{ 'submenu-title-noDropdown': !isNest }">
          <svg-icon v-if="item.children[0]?.meta?.icon"
            :icon-class="item.children[0]!.meta!.icon">
          </svg-icon>
          <template #title>
            <span v-if="item.children[0]?.meta?.title">{{ translateTitle(item.children[0]!.meta!.title) }}</span>
          </template>
        </el-menu-item>
      </router-link>
      <!-- Top-level menu with multiple children -->
      <el-sub-menu v-else :index="calcIndex(item)" :key="calcIndex(item)">
        <!-- Top-level menu -->
        <template #title>
          <svg-icon v-if="item.meta?.icon" :icon-class="item.meta.icon"></svg-icon>
          <span v-if="item.meta?.title">{{ translateTitle(item.meta.title) }}</span>
        </template>
        <!-- Submenu -->
        <template v-for="child in getFilteredChildren(item.children!)">
          <sidebar-item :is-nest="true" class="nest-menu" v-if="child.children && child.children.length > 0"
            :routes="[child as any]" :key="child.path"></sidebar-item>
          <!-- Submenu with external link -->
          <a v-else-if="child.path.startsWith('http')" :href="child.path" target="_blank" rel="noopener noreferrer"
            :aria-label="child.meta?.title ? translateTitle(child.meta.title) : undefined"
            :key="child.name || child.path">
            <el-menu-item :index="calcChildIndex(item, child)">
              <svg-icon v-if="child.meta?.icon" :icon-class="child.meta.icon"></svg-icon>
              <template #title>
                <span v-if="child.meta?.title">{{ translateTitle(child.meta.title) }}</span>
              </template>
            </el-menu-item>
          </a>
          <!-- Regular submenu -->
          <router-link v-else :to="calcChildIndex(item, child)" :key="'route-' + (child.name || child.path)">
            <el-menu-item :index="calcChildIndex(item, child)">
              <svg-icon v-if="child.meta?.icon" :icon-class="child.meta.icon"></svg-icon>
              <template #title>
                <span v-if="child.meta?.title">{{ translateTitle(child.meta.title) }}</span>
              </template>
            </el-menu-item>
          </router-link>
        </template>
      </el-sub-menu>

    </template>
  </div>
</template>
