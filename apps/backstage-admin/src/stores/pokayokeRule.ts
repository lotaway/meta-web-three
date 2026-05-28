import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { PokayokeRule, RuleStatus, RuleType } from '@/apis/pokayokeRule'
import {
  getRuleListAPI,
  getRuleByIdAPI,
  createRuleAPI,
  updateRuleAPI,
  deleteRuleAPI,
  activateRuleAPI,
  deactivateRuleAPI,
  getActiveRulesAPI,
  getRulesByWorkstationAPI,
} from '@/apis/pokayokeRule'

export const usePokayokeRuleStore = defineStore('pokayokeRule', () => {
  const rules = ref<PokayokeRule[]>([])
  const currentRule = ref<PokayokeRule | null>(null)
  const loading = ref(false)
  const total = ref(0)

  async function fetchRules(params?: { status?: RuleStatus; ruleType?: RuleType }) {
    loading.value = true
    try {
      const data = await getRuleListAPI(params)
      rules.value = data
      total.value = data.length
    } finally {
      loading.value = false
    }
  }

  async function fetchRuleById(id: number) {
    loading.value = true
    try {
      const data = await getRuleByIdAPI(id)
      currentRule.value = data
      return data
    } finally {
      loading.value = false
    }
  }

  async function createRule(rule: Partial<PokayokeRule>) {
    const data = await createRuleAPI(rule as any)
    rules.value.push(data)
    return data
  }

  async function updateRule(id: number, rule: Partial<PokayokeRule>) {
    const data = await updateRuleAPI(id, rule as any)
    const index = rules.value.findIndex(r => r.id === id)
    if (index !== -1) {
      rules.value[index] = data
    }
    if (currentRule.value?.id === id) {
      currentRule.value = data
    }
    return data
  }

  async function deleteRule(id: number) {
    await deleteRuleAPI(id)
    rules.value = rules.value.filter(r => r.id !== id)
    if (currentRule.value?.id === id) {
      currentRule.value = null
    }
  }

  async function activateRule(id: number) {
    await activateRuleAPI(id)
    const rule = rules.value.find(r => r.id === id)
    if (rule) {
      rule.status = 'ACTIVE'
    }
  }

  async function deactivateRule(id: number) {
    await deactivateRuleAPI(id)
    const rule = rules.value.find(r => r.id === id)
    if (rule) {
      rule.status = 'INACTIVE'
    }
  }

  async function fetchActiveRules() {
    const data = await getActiveRulesAPI()
    return data
  }

  async function fetchRulesByWorkstation(workstationId: string) {
    const data = await getRulesByWorkstationAPI(workstationId)
    return data
  }

  return {
    rules,
    currentRule,
    loading,
    total,
    fetchRules,
    fetchRuleById,
    createRule,
    updateRule,
    deleteRule,
    activateRule,
    deactivateRule,
    fetchActiveRules,
    fetchRulesByWorkstation,
  }
})