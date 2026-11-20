<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage, ElMessageBox, ElTree } from 'element-plus'
import { getMenuTreeListAPI } from '@/apis/menu.ts' // Updated path to apis
import { roleListMenuByRoleIdAPI, roleAllocMenuAPI } from '@/apis/role' // Updated path to apis
import type { TreeNodeData } from 'element-plus/es/components/tree/src/tree.type'
import type { UmsMenuNode } from '@/types/menu'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

// Get route params
const router = useRouter()
const route = useRoute()

// Current role ID
const roleId = ref<number>()
// All menu tree list
const menuTreeList = ref<UmsMenuNode[]>([])
// Create a ref to the tree component
const treeRef = ref<InstanceType<typeof ElTree>>()
// Define tree component properties
const defaultProps = {
  children: 'children',
  label: 'title'
}
// Get menu tree list
const treeList = async () => {
  const res = await getMenuTreeListAPI()
  menuTreeList.value = res.data
}
// Get menus for a role
const getRoleMenu = async () => {
  const res = await roleListMenuByRoleIdAPI(roleId.value!)
  const menuList = res.data
  const checkedMenuIds = menuList.filter(item => item.parentId !== 0).map(item => item.id!)
  treeRef.value!.setCheckedKeys(checkedMenuIds)
}

// Run on page creation
onMounted(() => {
  roleId.value = Number(route.query.roleId)
  treeList()
  getRoleMenu()
})

// Save menu assignment
const handleSave = async () => {
  const checkedNodes = treeRef.value!.getCheckedNodes() as TreeNodeData[]
  const checkedMenuIds = new Set<number>()
  checkedNodes.forEach(item => {
    checkedMenuIds.add(item.id)
    if (item.parentId !== 0) {
      checkedMenuIds.add(item.parentId)
    }
  })
  await ElMessageBox.confirm(t('dialog.confirmAssignMenu'), t('dialog.title'), {
    confirmButtonText: t('common.confirm'),
    cancelButtonText: t('common.cancel'),
    type: 'warning'
  })
  await roleAllocMenuAPI({ roleId: roleId.value!, menuIds: Array.from(checkedMenuIds).join(',') })
  ElMessage({
    message: '分配成功',
    type: 'success',
    duration: 1000
  })
  router.back()
}

// Clear selected items
const handleClear = () => {
  treeRef.value!.setCheckedKeys([])
}
</script>

<template>
  <el-card class="form-container" shadow="never">
    <el-tree ref="treeRef" :data="menuTreeList" show-checkbox default-expand-all node-key="id" :props="defaultProps"
      highlight-current>
    </el-tree>
    <div style="margin-top: 20px" align="center">
      <el-button type="primary" @click="handleSave()">保存</el-button>
      <el-button @click="handleClear()">清空</el-button>
    </div>
  </el-card>
</template>

<style scoped></style>
