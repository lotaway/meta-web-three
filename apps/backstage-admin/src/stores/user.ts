import { defineStore } from 'pinia'
import { getAdminInfoAPI, adminLoginAPI, adminLogoutAPI } from '@/apis/admin'
import { ref } from 'vue'
import type { LoginParam, UserInfo } from '@/types/admin'
import { USER_PERSIST_KEY, purgePersistedPassword } from '@/utils/user-storage'

export const useUserStore = defineStore(
  'user',
  () => {
    // User info
    const userInfo = ref<UserInfo>({
      username: '',
      avatar: '',
      roles: [],
      token: '',
      menus: [],
    })

    // User login
    const userLogin = async (loginParam: LoginParam) => {
      const res = await adminLoginAPI(loginParam)
      const tokenStr = res.data.tokenHead + res.data.token
      userInfo.value.token = tokenStr
      userInfo.value.username = loginParam.username
      await getUserInfo()
    }

    // Get user info
    const getUserInfo = async () => {
      const res = await getAdminInfoAPI()
      if (res.data.roles && res.data.roles.length > 0) {
        // Check whether the returned roles is a non-empty array
        userInfo.value.roles = res.data.roles
      } else {
        throw new Error('No role assigned to this user. Please assign a role first.')
      }
      userInfo.value.menus = res.data.menus
      userInfo.value.avatar = res.data.icon
    }

    // User logout
    const userLogout = async () => {
      await adminLogoutAPI()
      userInfo.value.token = ''
      userInfo.value.roles = []
    }

    // Frontend logout
    const fedLogout = () => {
      userInfo.value.token = ''
    }

    return {
      userInfo,
      userLogin,
      getUserInfo,
      userLogout,
      fedLogout,
    }
  },
  {
    // Persist only non-sensitive fields; drop credentials left by older builds.
    persist: {
      key: USER_PERSIST_KEY,
      pick: [
        'userInfo.username',
        'userInfo.avatar',
        'userInfo.roles',
        'userInfo.token',
        'userInfo.menus',
      ],
      beforeHydrate: () => {
        if (typeof localStorage !== 'undefined') purgePersistedPassword(localStorage)
      },
    },
  },
)
