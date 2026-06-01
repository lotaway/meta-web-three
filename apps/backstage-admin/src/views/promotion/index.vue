<script setup lang="ts">
import { ref } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { Ticket, Timer, Goods, Document, Present } from '@element-plus/icons-vue'

const router = useRouter()
const route = useRoute()

const activeMenu = ref(route.query.tab?.toString() || 'coupon')

const menuItems = [
  { key: 'coupon', label: 'promotion.couponManagement', icon: Ticket, path: '/promotion?tab=coupon' },
  { key: 'flash', label: 'promotion.flashPromotion', icon: Timer, path: '/promotion/flash' },
  { key: 'brand', label: 'promotion.homeBrand', icon: Goods, path: '/promotion/brand' },
  { key: 'subject', label: 'promotion.homeSubject', icon: Document, path: '/promotion/subject' },
  { key: 'newProduct', label: 'promotion.homeNewProduct', icon: Present, path: '/promotion/newProduct' }
]

const handleMenuClick = (item: typeof menuItems[0]) => {
  activeMenu.value = item.key
  router.push(item.path)
}
</script>

<template>
  <div class="promotion-layout">
    <el-container>
      <el-aside width="200px">
        <el-menu :default-active="activeMenu" @select="(key) => activeMenu = key">
          <el-menu-item v-for="item in menuItems" :key="item.key" :index="item.key" @click="handleMenuClick(item)">
            <el-icon><component :is="item.icon" /></el-icon>
            <span>{{ $t(item.label) }}</span>
          </el-menu-item>
        </el-menu>
      </el-aside>
      <el-main>
        <router-view />
      </el-main>
    </el-container>
  </div>
</template>

<style scoped lang="scss">
.promotion-layout {
  height: 100%;
  
  .el-container {
    height: 100%;
  }
  
  .el-aside {
    background-color: var(--el-bg-color);
    border-right: 1px solid var(--el-border-color);
  }
  
  .el-main {
    padding: 0;
    background-color: var(--el-bg-color);
  }
}
</style>