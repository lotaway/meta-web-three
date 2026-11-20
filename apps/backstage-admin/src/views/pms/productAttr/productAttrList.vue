<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { getProductAttributeListAPI, productAttributeDeleteByIds } from '@/apis/productAttr'
import { Tickets } from '@element-plus/icons-vue'
import type { PmsProductAttribute } from '@/types/productAttr'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

// Get route object
const router = useRouter()

// Attribute category ID
const cid = Number(router.currentRoute.value.query.cid)
// Attribute type
const type = Number(router.currentRoute.value.query.type)
// List query data
const listQuery = ref({
  pageNum: 1,
  pageSize: 5,
  type: type
})
const list = ref<PmsProductAttribute[]>([])
// List data
const total = ref(0)
// Loading state
const listLoading = ref(true)
// Get list data
const getList = async () => {
  listLoading.value = true
  try {
    const response = await getProductAttributeListAPI(cid, listQuery.value)
    listLoading.value = false
    list.value = response.data.list
    total.value = response.data.total
  } catch (error) {
    listLoading.value = false
    console.error(error)
  }
}

// Run after component mounts
onMounted(() => {
  getList()
})

// Current batch operation type
const operateType = ref<string>()
// Batch operation selected items
const multipleSelection = ref<PmsProductAttribute[]>([])
// Batch operation type set
const operates = ref([
  {
    label: "删除",
    value: "deleteProductAttr"
  }
])

// Add product attribute
const addProductAttr = () => {
  router.push({ path: '/pms/addProductAttr', query: { cid: router.currentRoute.value.query.cid, type: router.currentRoute.value.query.type } })
}

// Handle table selection state change
const handleSelectionChange = (val: PmsProductAttribute[]) => {
  multipleSelection.value = val
}

// Handle batch operations
const handleBatchOperate = async () => {
  if (multipleSelection.value.length < 1) {
    ElMessage.warning(t('message.selectOneRecord'))
    return
  }
  if (operateType.value !== 'deleteProductAttr') {
    ElMessage.warning(t('message.selectOperationType'))
    return
  }
  await handleDeleteProductAttr(multipleSelection.value.map(item => item.id!))
}

// Handle page size change
const handleSizeChange = (val: number) => {
  listQuery.value.pageNum = 1
  listQuery.value.pageSize = val
  getList()
}

// Handle current page change
const handleCurrentChange = (val: number) => {
  listQuery.value.pageNum = val
  getList()
}

// Handle update operation
const handleUpdate = (index: number, row: PmsProductAttribute) => {
  router.push({ path: '/pms/updateProductAttr', query: { id: row.id } })
}

// Handle delete product attribute
const handleDeleteProductAttr = async (ids: number[]) => {
  try {
    await ElMessageBox.confirm(t('dialog.confirmDeleteAttribute'), t('dialog.title'), {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
      type: 'warning'
    })
    await productAttributeDeleteByIds({ ids: ids.join(",") })
    ElMessage.success(t('common.deleteSuccess'))
    getList()
  } catch (error) {
    console.error(error)
  }
}

// Handle delete operation
const handleDelete = async (index: number, row: PmsProductAttribute) => {
  await handleDeleteProductAttr([row.id!])
}

// Attribute value input method
const inputTypeFilter = (value: number) => {
  if (value === 1) {
    return '从列表中选取'
  } else {
    return '手工录入'
  }
}

// Whether the attribute is optional
const selectTypeFilter = (value: number) => {
  if (value === 1) {
    return '单选'
  } else if (value === 2) {
    return '多选'
  } else {
    return '唯一'
  }
}
</script>

<template>
  <div class="app-container">
    <el-card class="operate-container" shadow="never">
      <el-icon class="el-icon-middle">
        <Tickets />
      </el-icon>
      <span>{{ t('role.page.dataList') }}</span>
      <el-button class="btn-add" @click="addProductAttr()">
        {{ t('common.add') }}
      </el-button>
    </el-card>
    <div class="table-container">
      <el-table ref="productAttrTable" :data="list" style="width: 100%" @selection-change="handleSelectionChange"
        v-loading="listLoading" border>
        <el-table-column type="selection" width="60" align="center"></el-table-column>
        <el-table-column :label="t('common.id')" width="100" align="center">
          <template #default="scope">{{ scope.row.id }}</template>
        </el-table-column>
        <el-table-column label="属性名称" width="140" align="center">
          <template #default="scope">{{ scope.row.name }}</template>
        </el-table-column>
        <el-table-column label="商品类型" width="140" align="center">
          <template>{{ $route.query.cname }}</template>
        </el-table-column>
        <el-table-column label="属性是否可选" width="120" align="center">
          <template #default="scope">{{ selectTypeFilter(scope.row.selectType) }}</template>
        </el-table-column>
        <el-table-column label="属性值的录入方式" width="150" align="center">
          <template #default="scope">{{ inputTypeFilter(scope.row.inputType) }}</template>
        </el-table-column>
        <el-table-column label="可选值列表" align="center">
          <template #default="scope">{{ scope.row.inputList }}</template>
        </el-table-column>
        <el-table-column label="排序" width="100" align="center">
          <template #default="scope">{{ scope.row.sort }}</template>
        </el-table-column>
        <el-table-column label="操作" width="200" align="center">
          <template #default="scope">
            <el-button size="small" @click="handleUpdate(scope.$index, scope.row as PmsProductAttribute)">{{ t('common.edit') }}
            </el-button>
            <el-button size="small" type="danger" @click="handleDelete(scope.$index, scope.row as PmsProductAttribute)">{{ t('common.delete') }}
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>
    <div class="batch-operate-container">
      <el-select v-model="operateType" placeholder="批量操作">
        <el-option v-for="item in operates" :key="item.value" :label="item.label" :value="item.value">
        </el-option>
      </el-select>
      <el-button style="margin-left: 20px" class="search-button" @click="handleBatchOperate()" type="primary">
        确定
      </el-button>
    </div>
    <div class="pagination-container">
      <el-pagination background @size-change="handleSizeChange" @current-change="handleCurrentChange"
        layout="total, sizes,prev, pager, next,jumper" :page-size="listQuery.pageSize" :page-sizes="[5, 10, 15]"
        v-model:current-page="listQuery.pageNum" :total="total">
      </el-pagination>
    </div>
  </div>
</template>

<style rel="stylesheet/scss" lang="scss" scoped></style>
