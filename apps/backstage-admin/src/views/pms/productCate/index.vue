<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Tickets } from '@element-plus/icons-vue'
import { getProductCategoryListAPI, productCategoryDeleteByIdAPI, productCategoryUpdateShowStatusAPI, productCategoryUpdateNavStatusAPI } from '@/apis/productCate'
import type { PmsProductCategory } from '@/types/productCate'

// Get route
const router = useRouter()
const route = useRoute()

// Current list page parent category ID
const parentId = ref(0)
// List query params
const listQuery = ref({
  pageNum: 1,
  pageSize: 10
})
// List data
const list = ref<PmsProductCategory[]>([])
// Total count
const total = ref(0)
// Loading state
const listLoading = ref(true)
// Reset parent ID
const resetParentId = () => {
  listQuery.value.pageNum = 1
  if (route.query.parentId != null) {
    parentId.value = Number(route.query.parentId)
  } else {
    parentId.value = 0
  }
}
// Get list data
const getList = async () => {
  listLoading.value = true
  const res = await getProductCategoryListAPI(parentId.value, listQuery.value)
  listLoading.value = false
  list.value = res.data.list
  total.value = res.data.total
}

// Run after component mounts
onMounted(() => {
  resetParentId()
  getList()
})

// Watch query param changes
watch(() => route.query, () => {
  resetParentId()
  getList()
})

// Add product category
const handleAddProductCate = () => {
  router.push('/pms/addProductCate')
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

// Handle navigation state change
const handleNavStatusChange = async (index: number, row: PmsProductCategory) => {
  await productCategoryUpdateNavStatusAPI({ ids: [row.id!].join(','), navStatus: row.navStatus })
  ElMessage({
    message: '修改成功',
    type: 'success',
    duration: 1000
  })
}

// Handle display status change
const handleShowStatusChange = async (index: number, row: PmsProductCategory) => {
  await productCategoryUpdateShowStatusAPI({ ids: [row.id!].join(','), showStatus: row.navStatus })
  ElMessage({
    message: '修改成功',
    type: 'success',
    duration: 1000
  })
}

// View child categories
const handleShowNextLevel = (index: number, row: PmsProductCategory) => {
  router.push({ path: '/pms/productCate', query: { parentId: row.id } })
}

// Transfer products
const handleTransferProduct = (index: number, row: PmsProductCategory) => {
  console.log('handleAddProductCate', row)
}

// Update category
const handleUpdate = (index: number, row: PmsProductCategory) => {
  router.push({ path: '/pms/updateProductCate', query: { id: row.id } })
}

// Delete category
const handleDelete = async (index: number, row: PmsProductCategory) => {
  await ElMessageBox.confirm('是否要删除该品牌', '提示', {
    confirmButtonText: '确定',
    cancelButtonText: '取消',
    type: 'warning'
  })
  await productCategoryDeleteByIdAPI(row.id!)
  ElMessage({
    message: '删除成功',
    type: 'success',
    duration: 1000
  })
  getList()
}

// Category level filter
const levelFilter = (value: number) => {
  if (value === 0) {
    return '一级'
  } else if (value === 1) {
    return '二级'
  }
}

// Disable child button
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
      <span>数据列表</span>
      <el-button class="btn-add" @click="handleAddProductCate()">
        添加
      </el-button>
    </el-card>
    <div class="table-container">
      <el-table ref="productCateTable" style="width: 100%" :data="list" v-loading="listLoading" border>
        <el-table-column label="编号" width="100" align="center">
          <template #default="scope">{{ scope.row.id }}</template>
        </el-table-column>
        <el-table-column label="分类名称" align="center">
          <template #default="scope">{{ scope.row.name }}</template>
        </el-table-column>
        <el-table-column label="级别" width="100" align="center">
          <template #default="scope">{{ levelFilter(scope.row.level) }}</template>
        </el-table-column>
        <el-table-column label="商品数量" width="100" align="center">
          <template #default="scope">{{ scope.row.productCount }}</template>
        </el-table-column>
        <el-table-column label="数量单位" width="100" align="center">
          <template #default="scope">{{ scope.row.productUnit }}</template>
        </el-table-column>
        <el-table-column label="导航栏" width="100" align="center">
          <template #default="scope">
            <el-switch @change="handleNavStatusChange(scope.$index, scope.row as PmsProductCategory)" :active-value="1" :inactive-value="0"
              v-model="scope.row.navStatus">
            </el-switch>
          </template>
        </el-table-column>
        <el-table-column label="是否显示" width="100" align="center">
          <template #default="scope">
            <el-switch @change="handleShowStatusChange(scope.$index, scope.row as PmsProductCategory)" :active-value="1" :inactive-value="0"
              v-model="scope.row.showStatus">
            </el-switch>
          </template>
        </el-table-column>
        <el-table-column label="排序" width="100" align="center">
          <template #default="scope">{{ scope.row.sort }}</template>
        </el-table-column>
        <el-table-column label="设置" width="200" align="center">
          <template #default="scope">
            <el-button size="small" :disabled="disableNextLevel(scope.row.level)"
              @click="handleShowNextLevel(scope.$index, scope.row as PmsProductCategory)">查看下级
            </el-button>
            <el-button size="small" @click="handleTransferProduct(scope.$index, scope.row as PmsProductCategory)">转移商品
            </el-button>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="200" align="center">
          <template #default="scope">
            <el-button size="small" @click="handleUpdate(scope.$index, scope.row as PmsProductCategory)">编辑
            </el-button>
            <el-button size="small" type="danger" @click="handleDelete(scope.$index, scope.row as PmsProductCategory)">删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>
    <div class="pagination-container">
      <el-pagination background @size-change="handleSizeChange" @current-change="handleCurrentChange"
        layout="total, sizes,prev, pager, next,jumper" :page-size="listQuery.pageSize" :page-sizes="[5, 10, 15]"
        v-model:current-page="listQuery.pageNum" :total="total">
      </el-pagination>
    </div>
  </div>
</template>

<style scoped></style>
