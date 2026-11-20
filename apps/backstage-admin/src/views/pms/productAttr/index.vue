<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { getProductAttributeCategoryListAPI, productAttributeCategoryCreateAPI, productAttributeCategoryDeleteById, productAttributeCategoryUpdateAPI } from '@/apis/productAttrCate'
import { Tickets } from '@element-plus/icons-vue'
import type { FormInstance } from 'element-plus'
import type { PageParam } from '@/types/common'
import type { PmsProductAttributeCategory } from '@/types/productAttr'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

// Get route object
const router = useRouter()

// List query params
const listQuery = ref<PageParam>({
  pageNum: 1,
  pageSize: 10
})
// List data
const list = ref<PmsProductAttributeCategory[]>([])
// Total count
const total = ref(0)
// Loading state
const listLoading = ref(true)
// Get list data
const getList = async () => {
  listLoading.value = true
  try {
    const response = await getProductAttributeCategoryListAPI(listQuery.value)
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

// Current attribute category
const productAttrCate = ref<PmsProductAttributeCategory>({
  name: ''
})
// Whether the edit dialog is visible
const dialogVisible = ref(false)
// Edit dialog title
const dialogTitle = ref('')
// Form ref in the edit dialog
const productAttrCatForm = ref<FormInstance>()
// Validation rules in the edit dialog
const rules = ref({
  name: [
    { required: true, message: '请输入类型名称', trigger: 'blur' }
  ]
})

// Add product attribute category
const addProductAttrCate = () => {
  dialogVisible.value = true
  dialogTitle.value = "添加类型"
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

// Handle delete operation
const handleDelete = async (index: number, row: PmsProductAttributeCategory) => {
  await ElMessageBox.confirm(t('dialog.confirmDeleteBrand'), t('dialog.title'), {
    confirmButtonText: t('common.confirm'),
    cancelButtonText: t('common.cancel'),
    type: 'warning'
  })
  await productAttributeCategoryDeleteById(row.id!)
  ElMessage.success(t('common.deleteSuccess'))
  getList()
}

// Handle update operation
const handleUpdate = (index: number, row: PmsProductAttributeCategory) => {
  dialogVisible.value = true
  dialogTitle.value = "编辑类型"
  productAttrCate.value.name = row.name
  productAttrCate.value.id = row.id
}

// Get attribute list
const getAttrList = (index: number, row: PmsProductAttributeCategory) => {
  router.push({ path: '/pms/productAttrList', query: { cid: row.id, cname: row.name, type: 0 } })
}

// Get parameter list
const getParamList = (index: number, row: PmsProductAttributeCategory) => {
  router.push({ path: '/pms/productAttrList', query: { cid: row.id, cname: row.name, type: 1 } })
}

// Confirm dialog operation
const handleConfirm = async () => {
  if (!productAttrCatForm.value) return
  const valid = await productAttrCatForm.value.validate().catch(() => false)
  if (valid) {
    if (dialogTitle.value === "添加类型") {
      await productAttributeCategoryCreateAPI(productAttrCate.value.name)
      ElMessage.success(t('message.operationSuccess'))
      dialogVisible.value = false
    } else {
      await productAttributeCategoryUpdateAPI(productAttrCate.value.id!, productAttrCate.value.name)
      ElMessage.success(t('message.modifySuccess'))
      dialogVisible.value = false
    }
    getList()
  } else {
    ElMessage.error(t('message.enterTypeName'))
    return false
  }
}

// Handle close dialog
const handleClose = () => {
  dialogVisible.value = false
}
</script>

<template>
  <div class="app-container">
    <el-card class="operate-container" shadow="never">
      <el-icon class="el-icon-middle">
        <Tickets />
      </el-icon>
      <span>{{ t('role.page.dataList') }}</span>
      <el-button class="btn-add" @click="addProductAttrCate()">
        {{ t('common.add') }}
      </el-button>
    </el-card>
    <div class="table-container">
      <el-table ref="productAttrCateTable" style="width: 100%" :data="list" v-loading="listLoading" border>
        <el-table-column :label="t('common.id')" width="100" align="center">
          <template #default="scope">{{ scope.row.id }}</template>
        </el-table-column>
        <el-table-column label="类型名称" align="center">
          <template #default="scope">{{ scope.row.name }}</template>
        </el-table-column>
        <el-table-column label="属性数量" width="200" align="center">
          <template #default="scope">{{ scope.row.attributeCount == null ? 0 : scope.row.attributeCount }}</template>
        </el-table-column>
        <el-table-column label="参数数量" width="200" align="center">
          <template #default="scope">{{ scope.row.paramCount == null ? 0 : scope.row.paramCount }}</template>
        </el-table-column>
        <el-table-column label="设置" width="200" align="center">
          <template #default="scope">
            <el-button size="small" @click="getAttrList(scope.$index, scope.row as PmsProductAttributeCategory)">{{ t('product.attrList') }}
            </el-button>
            <el-button size="small" @click="getParamList(scope.$index, scope.row as PmsProductAttributeCategory)">{{ t('product.paramList') }}
            </el-button>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="200" align="center">
          <template #default="scope">
            <el-button size="small" @click="handleUpdate(scope.$index, scope.row as PmsProductAttributeCategory)">{{ t('common.edit') }}
            </el-button>
            <el-button size="small" type="danger" @click="handleDelete(scope.$index, scope.row as PmsProductAttributeCategory)">{{ t('common.delete') }}
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
    <el-dialog :title="dialogTitle" v-model="dialogVisible" :before-close="handleClose" width="30%">
      <el-form ref="productAttrCatForm" :model="productAttrCate" :rules="rules" label-width="120px">
        <el-form-item label="类型名称" prop="name">
          <el-input v-model="productAttrCate.name" autocomplete="off"></el-input>
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="dialogVisible = false">{{ t('common.cancel') }}</el-button>
          <el-button type="primary" @click="handleConfirm">{{ t('common.confirm') }}</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<style rel="stylesheet/scss" lang="scss" scoped></style>
