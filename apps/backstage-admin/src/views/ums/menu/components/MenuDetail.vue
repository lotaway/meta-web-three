<script lang="ts" setup>
import { ref, reactive, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox, type FormInstance, type FormRules } from 'element-plus'
import { getMenuListByParentIdAPI, menuCreateAPI, updateMenu, getMenuByIdAPI } from '@/apis/menu.ts'
import type { UmsMenu } from '@/types/menu'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

// Get route object
const route = useRoute()
const router = useRouter()

// Define properties
const props = defineProps({
  isEdit: {
    type: Boolean,
    default: false
  }
})

// Default menu object
const defaultMenu: UmsMenu = {
  title: '',
  parentId: 0,
  name: '',
  icon: '',
  hidden: 0,
  sort: 0
}

// Menu data
const menu = ref(Object.assign({}, defaultMenu))
// Select menu list
const selectMenuList = ref<UmsMenu[]>([])
// Menu form component ref
const menuFromRef = ref<FormInstance>()
// Menu form validation rules
const rules = reactive<FormRules>({
  title: [
    { required: true, message: '请输入菜单名称', trigger: 'blur' },
    { min: 2, max: 140, message: '长度在 2 到 140 个字符', trigger: 'blur' }
  ],
  name: [
    { required: true, message: '请输入前端名称', trigger: 'blur' },
    { min: 2, max: 140, message: '长度在 2 到 140 个字符', trigger: 'blur' }
  ],
  icon: [
    { required: true, message: '请输入前端图标', trigger: 'blur' },
    { min: 2, max: 140, message: '长度在 2 到 140 个字符', trigger: 'blur' }
  ]
})

// Get select menu list
const getSelectMenuList = async () => {
  const res = await getMenuListByParentIdAPI(0, { pageSize: 100, pageNum: 1 })
  selectMenuList.value = res.data.list
  const noParentMenu = { ...defaultMenu }
  noParentMenu.id = 0
  noParentMenu.title = '无上级菜单'
  selectMenuList.value.unshift(noParentMenu)
}

// Load data on mount
onMounted(async () => {
  if (props.isEdit) {
    const res = await getMenuByIdAPI(Number(route.query.id))
    menu.value = res.data
  } else {
    menu.value = Object.assign({}, defaultMenu)
  }
  getSelectMenuList()
})

const saveMenu = async () => {
  if (props.isEdit) {
    await updateMenu(Number(route.query.id), menu.value)
    menuFromRef.value!.resetFields()
    return '修改成功'
  }
  await menuCreateAPI(menu.value)
  menuFromRef.value!.resetFields()
  resetForm()
  return '提交成功'
}

// Handle menu form submission
const onSubmit = () => {
  menuFromRef.value!.validate(async (valid) => {
    if (!valid) {
      ElMessage({ message: '验证失败', type: 'error', duration: 1000 })
      return
    }
    await ElMessageBox.confirm(t('dialog.confirmSubmit'), t('dialog.title'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    ElMessage({ message: await saveMenu(), type: 'success', duration: 1000 })
    router.back()
  })
}

// Handle menu form reset
const resetForm = () => {
  menuFromRef.value!.resetFields()
  menu.value = Object.assign({}, defaultMenu)
  getSelectMenuList()
}
</script>

<template>
  <el-card class="form-container" shadow="never">
    <el-form :model="menu" :rules="rules" ref="menuFromRef" label-width="150px">
      <el-form-item :label="t('ums.menuName')" prop="title">
        <el-input v-model="menu.title"></el-input>
      </el-form-item>
      <el-form-item :label="t('ums.parentMenu')">
        <el-select v-model="menu.parentId" :placeholder="t('ums.selectMenu')">
          <el-option v-for="item in selectMenuList" :key="item.id" :label="item.title" :value="item.id!">
          </el-option>
        </el-select>
      </el-form-item>
      <el-form-item :label="t('ums.frontName')" prop="name">
        <el-input v-model="menu.name"></el-input>
      </el-form-item>
      <el-form-item :label="t('ums.frontIcon')" prop="icon">
        <el-input v-model="menu.icon" style="width: 80%"></el-input>
        <svg-icon style="margin-left: 8px" :icon-class="menu.icon"></svg-icon>
      </el-form-item>
      <el-form-item :label="t('product.isDisplay')">
        <el-radio-group v-model="menu.hidden">
          <el-radio :label="0">{{ t('role.page.yes') }}</el-radio>
          <el-radio :label="1">{{ t('role.page.no') }}</el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item :label="t('product.sort')">
        <el-input v-model="menu.sort"></el-input>
      </el-form-item>
      <el-form-item>
        <el-button type="primary" @click="onSubmit()">{{ t('common.submit') }}</el-button>
        <el-button v-if="!props.isEdit" @click="resetForm()">{{ t('common.reset') }}</el-button>
      </el-form-item>
    </el-form>
  </el-card>
</template>

<style scoped></style>
