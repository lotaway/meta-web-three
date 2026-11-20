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

// Handle menu form submission
const onSubmit = () => {
  menuFromRef.value!.validate(async (valid) => {
    if (valid) {
      await ElMessageBox.confirm(t('dialog.confirmSubmit'), t('dialog.title'), {
        confirmButtonText: t('common.confirm'),
        cancelButtonText: t('common.cancel'),
        type: 'warning'
      })
      if (props.isEdit) {
        await updateMenu(Number(route.query.id), menu.value)
        menuFromRef.value!.resetFields()
        ElMessage({
          message: '修改成功',
          type: 'success',
          duration: 1000
        })
        router.back()
      } else {
        await menuCreateAPI(menu.value)
        menuFromRef.value!.resetFields()
        resetForm()
        ElMessage({
          message: '提交成功',
          type: 'success',
          duration: 1000
        })
        router.back()
      }
    } else {
      ElMessage({
        message: '验证失败',
        type: 'error',
        duration: 1000
      })
    }
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
      <el-form-item label="菜单名称：" prop="title">
        <el-input v-model="menu.title"></el-input>
      </el-form-item>
      <el-form-item label="上级菜单：">
        <el-select v-model="menu.parentId" placeholder="请选择菜单">
          <el-option v-for="item in selectMenuList" :key="item.id" :label="item.title" :value="item.id!">
          </el-option>
        </el-select>
      </el-form-item>
      <el-form-item label="前端名称：" prop="name">
        <el-input v-model="menu.name"></el-input>
      </el-form-item>
      <el-form-item label="前端图标：" prop="icon">
        <el-input v-model="menu.icon" style="width: 80%"></el-input>
        <svg-icon style="margin-left: 8px" :icon-class="menu.icon"></svg-icon>
      </el-form-item>
      <el-form-item label="是否显示：">
        <el-radio-group v-model="menu.hidden">
          <el-radio :label="0">是</el-radio>
          <el-radio :label="1">否</el-radio>
        </el-radio-group>
      </el-form-item>
      <el-form-item label="排序：">
        <el-input v-model="menu.sort"></el-input>
      </el-form-item>
      <el-form-item>
        <el-button type="primary" @click="onSubmit()">提交</el-button>
        <el-button v-if="!props.isEdit" @click="resetForm()">重置</el-button>
      </el-form-item>
    </el-form>
  </el-card>
</template>

<style scoped></style>
