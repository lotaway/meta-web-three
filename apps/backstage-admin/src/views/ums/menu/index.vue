<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Tickets } from '@element-plus/icons-vue'
import { getMenuListByParentIdAPI, deleteMenuByIdAPI, menuUpdateHiddenByIdAPI } from '@/apis/menu.ts'
import type { UmsMenu } from '@/types/menu'
import type { PageParam } from '@/types/common'
import { t } from '@/locales'

const router = useRouter()
const route = useRoute()

const i18n = (key: string) => t(`menu.${key}`)

const listQuery = ref<PageParam>({
  pageNum: 1,
  pageSize: 10
})
const list = ref<UmsMenu[]>([])
const parentId = ref(0)
const total = ref<number>()
const listLoading = ref(true)
const resetParentId = () => {
  listQuery.value.pageNum = 1
  if (route.query.parentId) {
    parentId.value = Number(route.query.parentId)
  } else {
    parentId.value = 0
  }
}
const getList = async () => {
  listLoading.value = true
  try {
    const response = await getMenuListByParentIdAPI(parentId.value, listQuery.value)
    listLoading.value = false
    list.value = response.data.list
    total.value = response.data.total
  } catch (error) {
    listLoading.value = false
    ElMessage.error(t('menu.loadFailed'))
  }
}

onMounted(() => {
  resetParentId()
  getList()
})

const handleAddMenu = () => {
  router.push('/ums/addMenu')
}

const handleSizeChange = (val: number) => {
  listQuery.value.pageNum = 1
  listQuery.value.pageSize = val
  getList()
}

const handleCurrentChange = (val: number) => {
  listQuery.value.pageNum = val
  getList()
}

const handleHiddenChange = async (index: number, row: UmsMenu) => {
  await menuUpdateHiddenByIdAPI(row.id!, { hidden: row.hidden })
  ElMessage({
    message: i18n('updateSuccess'),
    type: 'success',
    duration: 1000
  })
}

const handleShowNextLevel = (index: number, row: UmsMenu) => {
  router.push({ path: '/ums/menu', query: { parentId: row.id } })
}

const handleUpdate = (index: number, row: UmsMenu) => {
  router.push({ path: '/ums/updateMenu', query: { id: row.id } })
}

const handleDelete = async (index: number, row: UmsMenu) => {
  await ElMessageBox.confirm(i18n('confirmDeleteMenu'), i18n('prompt'), {
    confirmButtonText: i18n('confirm'),
    cancelButtonText: i18n('cancel'),
    type: 'warning',
  })
  try {
    await deleteMenuByIdAPI(row.id!)
    ElMessage({
      message: i18n('deleteSuccess'),
      type: 'success',
      duration: 1000
    })
    getList()
  } catch (error) {
    ElMessage.error(t('menu.deleteFailed'))
  }
}

watch(() => route, () => {
  resetParentId()
  getList()
}, { deep: true })

const getMenuTitle = (title: string) => {
  return t(`menu.${title}`) || title
}

const levelFilter = (value: number) => {
  if (value === 0) {
    return i18n('level1')
  } else if (value === 1) {
    return i18n('level2')
  }
  return ''
}

const disableNextLevel = (value: number) => {
  if (value === 0) {
    return false
  } else {
    return true
  }
}
</script>

<template>
  <div class="app-container">
    <el-card class="operate-container" shadow="never">
      <el-icon class="el-icon-middle">
        <Tickets />
      </el-icon>
      <span>{{ i18n('dataList') }}</span>
      <el-button class="btn-add" @click="handleAddMenu()">
        {{ i18n('add') }}
      </el-button>
    </el-card>
    <div class="table-container">
      <el-table ref="menuTable" style="width: 100%" :data="list" v-loading="listLoading" border>
        <el-table-column label="{{ i18n('id') }}" width="100" align="center">
          <template #default="scope">{{ scope.row.id }}</template>
        </el-table-column>
        <el-table-column label="{{ i18n('menuName') }}" align="center">
          <template #default="scope">{{ getMenuTitle(scope.row.title) }}</template>
        </el-table-column>
        <el-table-column label="{{ i18n('menuLevel') }}" width="100" align="center">
          <template #default="scope">{{ levelFilter(scope.row.level) }}</template>
        </el-table-column>
        <el-table-column label="{{ i18n('frontName') }}" align="center">
          <template #default="scope">{{ scope.row.name }}</template>
        </el-table-column>
        <el-table-column label="{{ i18n('frontIcon') }}" width="100" align="center">
          <template #default="scope"><svg-icon :icon-class="scope.row.icon"></svg-icon></template>
        </el-table-column>
        <el-table-column label="{{ i18n('isShow') }}" width="100" align="center">
          <template #default="scope">
            <el-switch @change="handleHiddenChange(scope.$index, scope.row)" :active-value="0" :inactive-value="1"
              v-model="scope.row.hidden">
            </el-switch>
          </template>
        </el-table-column>
        <el-table-column label="{{ i18n('sort') }}" width="100" align="center">
          <template #default="scope">{{ scope.row.sort }}</template>
        </el-table-column>
        <el-table-column label="{{ i18n('setting') }}" width="120" align="center">
          <template #default="scope">
            <el-button type="primary" size="small" link :disabled="disableNextLevel(scope.row.level)"
              @click="handleShowNextLevel(scope.$index, scope.row)">{{ i18n('viewSubLevel') }}
            </el-button>
          </template>
        </el-table-column>
        <el-table-column label="{{ i18n('operation') }}" width="200" align="center">
          <template #default="scope">
            <el-button size="small" type="primary" link @click="handleUpdate(scope.$index, scope.row)">{{ i18n('edit') }}
            </el-button>
            <el-button size="small" type="primary" link @click="handleDelete(scope.$index, scope.row)">{{ i18n('delete') }}
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>
    <div class="pagination-container">
      <el-pagination background @size-change="handleSizeChange" @current-change="handleCurrentChange"
        layout="total, sizes,prev, pager, next,jumper" :page-size="listQuery.pageSize" :page-sizes="[10, 15, 20]"
        v-model:current-page="listQuery.pageNum" :total="total">
      </el-pagination>
    </div>
  </div>
</template>

<style scoped></style>
