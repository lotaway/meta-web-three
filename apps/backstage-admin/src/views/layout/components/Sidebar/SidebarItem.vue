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

// 安全转换名字为字符串
const safeName = (name: unknown): string => {
  if (typeof name === 'string') return name
  if (name && typeof name === 'object') return ''
  return String(name)
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

// 过滤出需要显示的路由
const filteredRoutes = computed(() => {
  return props.routes!.filter(item => !item.hidden && item.children)
})

// 过滤出需要显示的子路由
const getFilteredChildren = (children: RouteRecordExt[]) => {
  return children.filter(child => !child.hidden)
}

// 判断路由下方是否只有一个子路由
const hasOneShowingChildren = (children: RouteRecordExt[]) => {
  const showingChildren = children.filter(item => {
    return !item.hidden
  })
  if (showingChildren.length === 1) {
    return true
  }
  return false
}
</script>

<template>
  <div class="menu-wrapper">
    <template v-for="item in filteredRoutes">
      <!-- 只有一个子菜单的一级菜单 -->
      <router-link
        v-if="item.children && hasOneShowingChildren(item.children) && !item.children[0]!.children && !item.alwaysShow"
        :to="item.path + '/' + item.children[0]!.path" :key="item.children[0]!.name">
        <el-menu-item :index="item.path + '/' + item.children[0]!.path"
          :class="{ 'submenu-title-noDropdown': !isNest }">
          <svg-icon v-if="item.children[0]!.meta && item.children[0]!.meta.icon"
            :icon-class="item.children[0]!.meta.icon">
          </svg-icon>
          <template #title>
            <span v-if="item.children[0]!.meta && item.children[0]!.meta.title">{{ translateTitle(item.children[0]!.meta.title) }}</span>
          </template>
        </el-menu-item>
      </router-link>
      <!-- 有多个子菜单的一级菜单 -->
      <el-sub-menu v-else :index="(safeName(item.name) || item.path) as string" :key="safeName(item.name) as string">
        <!-- 一级菜单 -->
        <template #title>
          <svg-icon v-if="item.meta && item.meta.icon" :icon-class="item.meta.icon"></svg-icon>
          <span v-if="item.meta && item.meta.title">{{ translateTitle(item.meta.title) }}</span>
        </template>
        <!-- 子菜单 -->
        <template v-for="child in getFilteredChildren(item.children!)">
          <sidebar-item :is-nest="true" class="nest-menu" v-if="child.children && child.children.length > 0"
            :routes="[child]" :key="child.path"></sidebar-item>
          <!-- 具有外链功能的子菜单 -->
          <a v-else-if="child.path.startsWith('http')" v-bind:href="child.path" target="_blank" :key="(safeName(child.name) || child.path) as string">
            <el-menu-item :index="item.path + '/' + child.path">
              <svg-icon v-if="child.meta && child.meta.icon" :icon-class="child.meta.icon"></svg-icon>
              <template #title>
                <span v-if="child.meta && child.meta.title">{{ translateTitle(child.meta.title) }}</span>
              </template>
            </el-menu-item>
          </a>
          <!-- 普通子菜单 -->
          <router-link v-else :to="item.path + '/' + child.path" :key="('route-' + (safeName(child.name) || child.path)) as string">
            <el-menu-item :index="item.path + '/' + child.path">
              <svg-icon v-if="child.meta && child.meta.icon" :icon-class="child.meta.icon"></svg-icon>
              <template #title>
                <span v-if="child.meta && child.meta.title">{{ translateTitle(child.meta.title) }}</span>
              </template>
            </el-menu-item>
          </router-link>
        </template>
      </el-sub-menu>

    </template>
  </div>
</template>
