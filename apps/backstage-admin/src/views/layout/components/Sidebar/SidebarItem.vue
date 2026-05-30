<script lang="ts" setup>
import type { RouteRecordExt } from '@/types/router'
import { computed, type PropType } from 'vue'
import { useI18n } from 'vue-i18n'

// 定义组件名称
defineOptions({
  name: 'SidebarItem'
})

const { t } = useI18n()

// 翻译标题函数：如果是翻译key则翻译，否则直接显示
const translateTitle = (title: string | undefined) => {
  if (!title) return ''
  // 如果是翻译key格式（包含点号），则翻译
  if (title.includes('.') && !title.includes(' ')) {
    return t(title)
  }
  return title
}

// 安全的路径处理
const toStr = (v: unknown): string => {
  if (v === null || v === undefined) return ''
  if (typeof v === 'object') return ''
  return String(v)
}

// 构建路径
const joinPath = (...parts: unknown[]): string => {
  return parts.map(p => toStr(p)).filter(Boolean).join('/')
}

// 定义props
const props = defineProps({
  // 生成菜单的路由
  routes: {
    type: Array as PropType<RouteRecordExt[]>
  },
  // 控制只有一个子菜单的一级菜单样式
  isNest: {
    type: Boolean,
    default: false
  }
})

// 预处理路由数据，转换为字符串路径
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

// 判断路由下方是否只有一个子路由
const hasOneShowingChildren = (children: ProcessedRoute[]) => {
  const showingChildren = children.filter(item => !item.hidden)
  return showingChildren.length === 1
}

// 获取过滤后的子路由
const getFilteredChildren = (children: ProcessedRoute[]) => {
  return children.filter(child => !child.hidden)
}

// 计算菜单 index
const calcIndex = (item: ProcessedRoute): string => {
  return item.name || item.path
}

// 计算子路由 index
const calcChildIndex = (parent: ProcessedRoute, child: ProcessedRoute): string => {
  return joinPath(parent.path, child.path)
}
</script>

<template>
  <div class="menu-wrapper">
    <template v-for="item in processedRoutes">
      <!-- 只有一个子菜单的一级菜单 -->
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
      <!-- 有多个子菜单的一级菜单 -->
      <el-sub-menu v-else :index="calcIndex(item)" :key="calcIndex(item)">
        <!-- 一级菜单 -->
        <template #title>
          <svg-icon v-if="item.meta?.icon" :icon-class="item.meta.icon"></svg-icon>
          <span v-if="item.meta?.title">{{ translateTitle(item.meta.title) }}</span>
        </template>
        <!-- 子菜单 -->
        <template v-for="child in getFilteredChildren(item.children!)">
          <sidebar-item :is-nest="true" class="nest-menu" v-if="child.children && child.children.length > 0"
            :routes="[child as any]" :key="child.path"></sidebar-item>
          <!-- 具有外链功能的子菜单 -->
          <a v-else-if="child.path.startsWith('http')" :href="child.path" target="_blank" :key="child.name || child.path">
            <el-menu-item :index="calcChildIndex(item, child)">
              <svg-icon v-if="child.meta?.icon" :icon-class="child.meta.icon"></svg-icon>
              <template #title>
                <span v-if="child.meta?.title">{{ translateTitle(child.meta.title) }}</span>
              </template>
            </el-menu-item>
          </a>
          <!-- 普通子菜单 -->
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
