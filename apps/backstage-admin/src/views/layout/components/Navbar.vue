<script lang="ts" setup>
import { computed } from 'vue'
import Breadcrumb from '@/components/Breadcrumb/index.vue'
import Hamburger from '@/components/Hamburger/index.vue'
import { useAppStore } from '@/stores/app'
import { useUserStore } from '@/stores/user'
import { ArrowDown } from '@element-plus/icons-vue'
import { getLocale, setLocale, t, type LocaleType } from '@/locales'
import { ElMessage } from 'element-plus'

// Define component name
defineOptions({
  name: 'Navbar'
})

const appStore = useAppStore()
const userStore = useUserStore()

const sidebar = computed(() => appStore.sidebar)
const avatar = computed(() => userStore.userInfo.avatar)

// Current locale
const currentLang = computed(() => getLocale())

// i18n text
const homeText = computed(() => t('common.home'))
const logoutText = computed(() => t('common.logout'))

// Handle sidebar toggle
const handleToggleSideBar = () => {
  appStore.toggleSideBar()
}

// Handle user logout
const handleLogout = async () => {
  await userStore.userLogout()
  // Reload to reinitialize vue-router
  location.reload()
}

// Handle language switch
const handleSwitchLanguage = (lang: LocaleType) => {
  setLocale(lang)
  // Reload to apply new locale
  location.reload()
}
</script>

<template>
  <el-menu class="navbar" mode="horizontal">
    <hamburger class="hamburger-container" :toggle-click="handleToggleSideBar" :is-active="sidebar.opened"></hamburger>
    <breadcrumb></breadcrumb>
    
    <!-- Language Switcher -->
    <el-dropdown class="language-container" trigger="click" @command="handleSwitchLanguage">
      <div class="language-wrapper">
        <span class="language-text">{{ currentLang === 'zh-CN' ? '中文' : 'EN' }}</span>
        <el-icon class="el-icon-caret-bottom">
          <arrow-down />
        </el-icon>
      </div>
      <template #dropdown>
        <el-dropdown-menu>
          <el-dropdown-item :command="'zh-CN'" :disabled="currentLang === 'zh-CN'">
            中文
          </el-dropdown-item>
          <el-dropdown-item :command="'en-US'" :disabled="currentLang === 'en-US'">
            English
          </el-dropdown-item>
        </el-dropdown-menu>
      </template>
    </el-dropdown>
    
    <el-dropdown class="avatar-container" trigger="click">
      <div class="avatar-wrapper">
        <img class="user-avatar" :src="avatar">
        <el-icon class="el-icon-caret-bottom">
          <arrow-down />
        </el-icon>
      </div>
      <template #dropdown>
        <el-dropdown-menu class="user-dropdown">
          <router-link class="inlineBlock" to="/">
            <el-dropdown-item>
              {{ homeText }}
            </el-dropdown-item>
          </router-link>
          <el-dropdown-item divided>
            <span @click="handleLogout" style="display:block;">{{ logoutText }}</span>
          </el-dropdown-item>
        </el-dropdown-menu>
      </template>
    </el-dropdown>
  </el-menu>
</template>

<style lang="scss" scoped>
.navbar {
  height: 50px;
  line-height: 50px;
  border-radius: 0px !important;

  .hamburger-container {
    line-height: 58px;
    height: 50px;
    float: left;
    padding: 0 10px;
  }

  .screenfull {
    position: absolute;
    right: 90px;
    top: 16px;
    color: red;
  }

  .language-container {
    height: 50px;
    display: inline-block;
    position: absolute;
    right: 120px;
    cursor: pointer;

    .language-wrapper {
      display: flex;
      align-items: center;
      height: 50px;
      
      .language-text {
        font-size: 14px;
        margin-right: 5px;
      }

      .el-icon-caret-bottom {
        font-size: 12px;
      }
    }
  }

  .avatar-container {
    height: 50px;
    display: inline-block;
    position: absolute;
    right: 35px;

    .avatar-wrapper {
      cursor: pointer;
      margin-top: 5px;
      position: relative;

      .user-avatar {
        width: 40px;
        height: 40px;
        border-radius: 10px;
      }

      .el-icon-caret-bottom {
        position: absolute;
        right: -20px;
        top: 25px;
        font-size: 12px;
      }
    }
  }
}
</style>