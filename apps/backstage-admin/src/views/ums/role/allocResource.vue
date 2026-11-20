<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { fetchAllResourceList } from '@/apis/resource.ts'
import { resourceCategoryListAllAPI } from '@/apis/resourceCategory.ts'
import { roleAllocResourceAPI, roleListResourceById } from '@/apis/role'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { UmsResource, UmsResourceCategory } from '@/types/resource'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

// Get route
const router = useRouter()
const route = useRoute()

// Current role ID
const roleId = ref<number>()
// All resource list
const allResource = ref<UmsResource[]>([])
// All resource category list
const allResourceCate = ref<UmsResourceCategory[]>([])
// Get all resource categories
const getAllResourceCateList = async () => {
  const res = await resourceCategoryListAllAPI()
  allResourceCate.value = res.data
  allResourceCate.value.forEach(item => item.checked = false)
}
// Get all resources
const getAllResourceList = async () => {
  const res = await fetchAllResourceList()
  allResource.value = res.data
  allResource.value.forEach(item => item.checked = false)
}

// Get assigned resources by role and set selection state
const getResourceByRole = async () => {
  const res = await roleListResourceById(roleId.value!)
  const allocResource = res.data
  allResource.value.forEach(item => {
    item.checked = getResourceChecked(item.id!, allocResource)
  })
  allResourceCate.value.forEach(item => {
    item.checked = isAllChecked(item.id!)
  })
}

// Run on page mount
onMounted(async () => {
  roleId.value = Number(route.query.roleId)
  await getAllResourceCateList()
  await getAllResourceList()
  await getResourceByRole()
})

// Get resources by category ID
const getResourceByCate = (categoryId: number) => {
  return allResource.value.filter(item => item.categoryId === categoryId)
}

// Check whether a resource is selected
const getResourceChecked = (resourceId: number, allocResource: UmsResource[]) => {
  const index = allocResource.findIndex(item => item.id === resourceId)
  return index > -1
}

// Check whether a category is partially selected
const isIndeterminate = (categoryId: number) => {
  const cateResources = getResourceByCate(categoryId)
  return !(cateResources.every(item => item.checked === true) || cateResources.every(item => item.checked === false))
}

// Check whether a category is fully selected
const isAllChecked = (categoryId: number) => {
  const cateResources = getResourceByCate(categoryId)
  return cateResources.every(item => item.checked === true)
}

// Save resource assignment
const handleSave = async () => {
  await ElMessageBox.confirm(t('dialog.confirmAssignResource'), t('dialog.title'), {
    confirmButtonText: t('common.confirm'),
    cancelButtonText: t('common.cancel'),
    type: 'warning'
  })
  const checkedResourceIds = new Set<number>()
  if (allResource.value && allResource.value.length > 0) {
    allResource.value.forEach(item => {
      if (item.checked) {
        checkedResourceIds.add(item.id!)
      }
    })
  }
  await roleAllocResourceAPI({ roleId: roleId.value!, resourceIds: Array.from(checkedResourceIds).join(',') })
  ElMessage({
    message: '分配成功',
    type: 'success',
    duration: 1000
  })
  router.back()
}

// Clear selected items
const handleClear = () => {
  allResourceCate.value.forEach(item => {
    item.checked = false
  })
  allResource.value.forEach(item => {
    item.checked = false
  })
}

// Handle select-all change event
const handleCheckAllChange = (cate: UmsResourceCategory) => {
  const cateResources = getResourceByCate(cate.id!)
  cateResources.forEach(item => item.checked = cate.checked)
}

// Handle single resource selection event
const handleCheckChange = (resource: UmsResource) => {
  allResourceCate.value.forEach(item => {
    if (item.id === resource.categoryId) {
      item.checked = isAllChecked(resource.categoryId)
    }
  })
}
</script>

<template>
  <el-card class="form-container" shadow="never">
    <div v-for="(cate, index) in allResourceCate" :class="index === 0 ? 'top-line' : null" :key="'cate' + cate.id">
      <el-row class="table-layout" style="background: #F2F6FC;">
        <el-checkbox v-model="cate.checked" :indeterminate="isIndeterminate(cate.id!)"
          @change="handleCheckAllChange(cate)">
          {{ cate.name }}
        </el-checkbox>
      </el-row>
      <el-row class="table-layout">
        <el-col :span="8" v-for="resource in getResourceByCate(cate.id!)" :key="resource.id" style="padding: 4px 0">
          <el-checkbox v-model="resource.checked" @change="handleCheckChange(resource)">
            {{ resource.name }}
          </el-checkbox>
        </el-col>
      </el-row>
    </div>
    <div style="margin-top: 20px" align="center">
      <el-button type="primary" @click="handleSave()">保存</el-button>
      <el-button @click="handleClear()">清空</el-button>
    </div>
  </el-card>
</template>

<style scoped>
.table-layout {
  padding: 20px;
  border-left: 1px solid #DCDFE6;
  border-right: 1px solid #DCDFE6;
  border-bottom: 1px solid #DCDFE6;
}

.top-line {
  border-top: 1px solid #DCDFE6;
}
</style>
