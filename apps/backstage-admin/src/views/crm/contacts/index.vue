<template>
  <div class="crm-contacts">
    <h2>{{ t('crm.contact.title') }}</h2>
    <el-card class="search-bar">
      <el-form :inline="true" :model="queryParams">
        <el-form-item>
          <el-input v-model="queryParams.keywords" :placeholder="t('common.search')" clearable @keyup.enter="handleSearch" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleSearch">{{ t('common.search') }}</el-button>
          <el-button @click="handleReset">{{ t('common.reset') }}</el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="table-card">
      <div class="table-header">
        <el-button type="primary" @click="openCreate">{{ t('common.add') }}</el-button>
      </div>
      <el-table :data="contacts" stripe v-loading="loading">
        <el-table-column :label="t('crm.contact.name')" width="150">
          <template #default="{ row }">{{ row.firstName }} {{ row.lastName }}</template>
        </el-table-column>
        <el-table-column prop="email" :label="t('crm.contact.email')" />
        <el-table-column prop="phone" :label="t('crm.contact.phone')" width="120" />
        <el-table-column prop="position" :label="t('crm.contact.position')" />
        <el-table-column prop="department" :label="t('crm.contact.department')" />
        <el-table-column prop="customerId" :label="t('crm.contact.customer')" />
        <el-table-column prop="isPrimary" :label="t('crm.contact.isPrimary')" width="80">
          <template #default="{ row }">
            <el-tag :type="row.isPrimary ? 'success' : 'info'" size="small">{{ row.isPrimary ? t('common.yes') : t('common.no') }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column :label="t('common.operations')" width="160" fixed="right">
          <template #default="{ row }">
            <el-button size="small" type="primary" link @click="openEdit(row)">{{ t('common.edit') }}</el-button>
            <el-popconfirm :title="t('common.confirm') + t('common.delete') + '?'" @confirm="handleDelete(row.id!)">
              <template #reference>
                <el-button size="small" type="danger" link>{{ t('common.delete') }}</el-button>
              </template>
            </el-popconfirm>
          </template>
        </el-table-column>
      </el-table>
      <el-pagination
        v-model:current-page="page"
        :page-size="pageSize"
        :total="total"
        layout="total, prev, pager, next"
        @current-change="fetchContacts"
      />
    </el-card>

    <el-dialog v-model="dialogVisible" :title="isEdit ? t('crm.contact.edit') : t('crm.contact.add')" width="600px">
      <el-form ref="formRef" :model="form" :rules="rules" label-width="100px">
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('crm.contact.firstName')" prop="firstName">
              <el-input v-model="form.firstName" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('crm.contact.lastName')" prop="lastName">
              <el-input v-model="form.lastName" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('crm.contact.email')" prop="email">
              <el-input v-model="form.email" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('crm.contact.phone')" prop="phone">
              <el-input v-model="form.phone" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="12">
            <el-form-item :label="t('crm.contact.position')" prop="position">
              <el-input v-model="form.position" />
            </el-form-item>
          </el-col>
          <el-col :span="12">
            <el-form-item :label="t('crm.contact.department')" prop="department">
              <el-input v-model="form.department" />
            </el-form-item>
          </el-col>
        </el-row>
        <el-form-item :label="t('crm.contact.isPrimary')" prop="isPrimary">
          <el-switch v-model="form.isPrimary" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">{{ t('common.cancel') }}</el-button>
        <el-button type="primary" @click="handleSave">{{ t('common.save') }}</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>
import { ref, reactive, onMounted } from 'vue'
import { t } from '@/locales'
import { ElMessage } from 'element-plus'
import { listContacts, createContact, updateContact, deleteContact } from '@/apis/crm'
import type { Contact } from '@/apis/crm'

const contacts = ref<Contact[]>([])
const loading = ref(false)
const total = ref(0)
const page = ref(1)
const pageSize = ref(10)

const queryParams = reactive({
  keywords: '',
})

const dialogVisible = ref(false)
const isEdit = ref(false)
const editingId = ref<number | undefined>()

const form = reactive({
  firstName: '',
  lastName: '',
  email: '',
  phone: '',
  position: '',
  department: '',
  isPrimary: false,
})

const rules = {
  firstName: [{ required: true, message: t('crm.contact.firstName') + t('common.requiredSuffix'), trigger: 'blur' }],
  lastName: [{ required: true, message: t('crm.contact.lastName') + t('common.requiredSuffix'), trigger: 'blur' }],
}

const fetchContacts = async () => {
  loading.value = true
  try {
    const res = await listContacts({ ...queryParams, page: page.value, pageSize: pageSize.value })
    contacts.value = res.data.records
    total.value = res.data.total
  } catch (e) {
    console.error('Failed to fetch contacts:', e)
    ElMessage.error('Failed to load contacts')
  } finally {
    loading.value = false
  }
}

const handleSearch = () => {
  page.value = 1
  fetchContacts()
}

const handleReset = () => {
  queryParams.keywords = ''
  page.value = 1
  fetchContacts()
}

const openCreate = () => {
  isEdit.value = false
  editingId.value = undefined
  form.firstName = ''
  form.lastName = ''
  form.email = ''
  form.phone = ''
  form.position = ''
  form.department = ''
  form.isPrimary = false
  dialogVisible.value = true
}

const openEdit = (row: Contact) => {
  isEdit.value = true
  editingId.value = row.id
  form.firstName = row.firstName
  form.lastName = row.lastName
  form.email = row.email
  form.phone = row.phone
  form.position = row.position
  form.department = row.department
  form.isPrimary = row.isPrimary
  dialogVisible.value = true
}

const handleSave = async () => {
  try {
    if (isEdit.value && editingId.value) {
      await updateContact({ id: editingId.value, ...form })
      ElMessage.success(t('message.updateSuccess'))
    } else {
      await createContact(form)
      ElMessage.success(t('message.createSuccess'))
    }
    dialogVisible.value = false
    fetchContacts()
  } catch (e) {
    console.error('Failed to save contact:', e)
    ElMessage.error(t('message.operationFailed'))
  }
}

const handleDelete = async (id: number) => {
  try {
    await deleteContact(id)
    ElMessage.success(t('message.deleteSuccess'))
    fetchContacts()
  } catch (e) {
    console.error('Failed to delete contact:', e)
    ElMessage.error(t('message.operationFailed'))
  }
}

onMounted(fetchContacts)
</script>

<style scoped>
.crm-contacts { padding: 20px; }
.search-bar { margin-bottom: 16px; }
.table-header { margin-bottom: 16px; }
</style>
