<!-- frontend/src/App.vue -->
<template>
  <div id="app" class="app-container">
    <header class="app-header">
      <div class="header-content">
        <h1>🏥 医疗AI多智能体咨询系统</h1>
        <div class="header-controls">
          <div class="view-toggle">
            <button 
              :class="{ active: currentView === 'discussion' }" 
              @click="currentView = 'discussion'"
              class="toggle-btn"
            >
              💬 MDT讨论
            </button>
            <button 
              :class="{ active: currentView === 'chat' }" 
              @click="currentView = 'chat'"
              class="toggle-btn"
            >
              💭 传统聊天
            </button>
          </div>
          <div class="header-status">
            <span
              class="status-indicator"
              :class="{ connected: isConnected }"
            ></span>
            <span>{{ isConnected ? "已连接" : "未连接" }}</span>
          </div>
        </div>
      </div>
    </header>

    <div class="main-content">
      <!-- MDT讨论界面 -->
      <div v-if="currentView === 'discussion'" class="discussion-view">
        <!-- 智能体状态面板 -->
        <aside class="agents-panel">
          <h3>👥 专家团队</h3>
          <div class="agent-card" v-for="agent in agents" :key="agent.id">
            <div class="agent-avatar">{{ agent.emoji }}</div>
            <div class="agent-info">
              <h4>{{ agent.name }}</h4>
              <p class="agent-specialty">{{ agent.specialty }}</p>
              <div class="agent-status" :class="agent.status">
                {{ agent.statusText }}
              </div>
            </div>
          </div>
        </aside>

        <!-- 讨论流程主界面 -->
        <main class="discussion-main">
          <!-- 快速启动按钮 -->
          <div v-if="discussionMessages.length === 0 && !isLoading" class="quick-start-panel">
            <div class="welcome-message">
              <h2>🏥 欢迎使用MDT多学科团队讨论系统</h2>
              <p>点击下方按钮开始完整的7阶段MDT讨论流程：</p>
              <div class="flow-preview">
                📋 初始分析 → 👨‍⚕️ 肿瘤科专家 → 👩‍⚕️ 护理专家 → 🧠 心理专家 → 💬 团队讨论 → ✅ 达成共识 → 📋 生成方案
              </div>
              <button @click="simulateMDTDiscussion" class="start-discussion-btn">
                🚀 开始MDT讨论
              </button>
            </div>
          </div>

          <DiscussionFlow
            :messages="discussionMessages"
            :current-stage="discussionStage"
            :is-loading="isLoading"
            @scroll-to-bottom="scrollToBottom"
          />

          <!-- 患者方案展示 -->
          <PatientPlan
            v-if="finalPatientPlan"
            :plan="finalPatientPlan"
            @approve-plan="approvePlan"
            @request-modification="requestModification"
            @export-plan="exportPlan"
            @print-plan="printPlan"
          />

          <!-- 输入框 -->
          <InputBox
            @send-message="sendDiscussionMessage"
            :disabled="isLoading"
            :placeholder="discussionPlaceholder"
          />
        </main>

        <!-- 讨论进度面板 -->
        <aside class="progress-panel">
          <h3>📊 讨论进度</h3>
          <div class="progress-stages">
            <div 
              v-for="(stage, index) in discussionStages" 
              :key="stage.id"
              class="stage-item"
              :class="{ 
                active: stage.id === discussionStage, 
                completed: stage.completed 
              }"
            >
              <div class="stage-number">{{ index + 1 }}</div>
              <div class="stage-info">
                <div class="stage-name">{{ stage.name }}</div>
                <div class="stage-description">{{ stage.description }}</div>
              </div>
            </div>
          </div>

          <!-- 共识统计 -->
          <div class="consensus-stats" v-if="consensusStats">
            <h4>🤝 共识统计</h4>
            <div class="stats-grid">
              <div class="stat-item">
                <span class="stat-value">{{ consensusStats.agreement }}%</span>
                <span class="stat-label">一致性</span>
              </div>
              <div class="stat-item">
                <span class="stat-value">{{ consensusStats.confidence }}%</span>
                <span class="stat-label">置信度</span>
              </div>
            </div>
          </div>
        </aside>
      </div>

      <!-- 传统聊天界面 -->
      <div v-else class="chat-view">
        <!-- 智能体状态面板 -->
        <aside class="agents-panel">
          <h3>🤖 智能体团队</h3>
          <div class="agent-card" v-for="agent in agents" :key="agent.id">
            <div class="agent-avatar">{{ agent.emoji }}</div>
            <div class="agent-info">
              <h4>{{ agent.name }}</h4>
              <p class="agent-specialty">{{ agent.specialty }}</p>
              <div class="agent-status" :class="agent.status">
                {{ agent.statusText }}
              </div>
            </div>
          </div>
        </aside>

        <!-- 聊天窗口 -->
        <main class="chat-container">
          <ChatWindow
            :messages="messages"
            :is-loading="isLoading"
            @scroll-to-bottom="scrollToBottom"
          />

          <InputBox
            @send-message="sendMessage"
            :disabled="isLoading"
            :placeholder="inputPlaceholder"
          />
        </main>

        <!-- 建议面板 -->
        <aside class="recommendations-panel">
          <h3>💡 推荐建议</h3>
          <div
            v-if="currentRecommendations.length > 0"
            class="recommendations-list"
          >
            <div
              v-for="(rec, index) in currentRecommendations"
              :key="index"
              class="recommendation-item"
            >
              <span class="rec-number">{{ index + 1 }}</span>
              <span class="rec-text">{{ rec }}</span>
            </div>
          </div>
          <div v-else class="no-recommendations">
            <p>暂无推荐建议</p>
            <p class="hint">开始咨询后，专家团队会给出相应建议</p>
          </div>
        </aside>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, reactive, onMounted, onUnmounted, computed, nextTick } from "vue";
import ChatWindow from "./components/ChatWindow.vue";
import InputBox from "./components/InputBox.vue";
import DiscussionFlow from "./components/DiscussionFlow.vue";
import PatientPlan from "./components/PatientPlan.vue";
import axios from "axios";

// 创建axios实例
const apiClient = axios.create({
  baseURL: "http://localhost:8000",
  timeout: 600000, // 增加超时时间到600秒
  headers: {
    "Content-Type": "application/json",
  },
});

export default {
  name: "App",
  components: {
    ChatWindow,
    InputBox,
    DiscussionFlow,
    PatientPlan,
  },
  setup() {
    const isConnected = ref(true);
    const isLoading = ref(false);
    const messages = ref([]);
    const currentRecommendations = ref([]);
    const healthCheckInterval = ref(null);

    // 新增：界面切换和MDT讨论相关数据
    const currentView = ref('discussion'); // 'discussion' 或 'chat'
    const discussionMessages = ref([]);
    const discussionStage = ref('initial');
    const finalPatientPlan = ref(null);
    const consensusStats = ref(null);

    const agents = reactive([
      {
        id: "oncologist",
        name: "肿瘤科医生",
        emoji: "🩺",
        specialty: "肿瘤诊断与治疗",
        status: "idle",
        statusText: "待命中",
      },
      {
        id: "radiologist",
        name: "影像科医生",
        emoji: "🔍",
        specialty: "医学影像分析",
        status: "idle",
        statusText: "待命中",
      },
      {
        id: "nurse",
        name: "肿瘤专科护士",
        emoji: "👩",
        specialty: "护理管理与症状支持",
        status: "idle",
        statusText: "待命中",
      },
      {
        id: "patient_advocate",
        name: "患者权益倡导者",
        emoji: "👨",
        specialty: "权益保障与决策支持",
        status: "idle",
        statusText: "待命中",
      },
      {
        id: "psychologist",
        name: "肿瘤心理专家",
        emoji: "🧠",
        specialty: "心理评估与干预",
        status: "idle",
        statusText: "待命中",
      },
    ]);

    // 讨论阶段配置
    const discussionStages = reactive([
      {
        id: 'initial',
        name: '病例介绍',
        description: '患者基本信息和主诉',
        completed: false
      },
      {
        id: 'analysis',
        name: '专家分析',
        description: '各专家独立分析',
        completed: false
      },
      {
        id: 'discussion',
        name: '团队讨论',
        description: '专家间交流讨论',
        completed: false
      },
      {
        id: 'consensus',
        name: '达成共识',
        description: '形成统一意见',
        completed: false
      },
      {
        id: 'plan',
        name: '制定方案',
        description: '最终治疗方案',
        completed: false
      }
    ]);

    const inputPlaceholder = computed(() => {
      if (isLoading.value) return "专家正在分析中...";
      return "请描述您的症状或问题...";
    });

    const discussionPlaceholder = computed(() => {
      if (isLoading.value) return "专家团队正在讨论中...";
      
      const stageTexts = {
        'initial': '请详细描述患者的基本信息、症状和病史...',
        'analysis': '您可以补充更多信息，或输入"继续"进入下一阶段...',
        'discussion': '您可以提出问题或补充信息，或输入"继续"进入共识阶段...',
        'consensus': '您可以对专家意见提出看法，或输入"继续"制定方案...',
        'plan': '✅ 共识已达成！您可以就治疗方案提问、要求调整，或输入"继续"生成最终方案...',
        'completed': '🎉 治疗方案已制定完成！您可以继续提问、要求调整方案或咨询其他问题...'
      };
      
      return stageTexts[discussionStage.value] || "您可以继续提问或补充信息...";
    });

    // 健康检查函数 - 优化版本，增加错误处理和重试机制
    const checkHealth = async () => {
      try {
        // 重置连接状态为false，只有在成功响应后才设置为true
        isConnected.value = false;
        
        console.log('正在进行健康检查...');
        const response = await apiClient.get("/health");
        console.log('健康检查响应:', response.data);
        
        if (response.data.status === "healthy") {
          isConnected.value = true;

          // 更新智能体状态
          if (response.data.agents) {
            Object.entries(response.data.agents).forEach(
              ([agentId, status]) => {
                const agent = agents.find((a) => a.id === agentId);
                console.log(`更新智能体${agentId}状态为${status}`)
                if (agent) {
                  agent.status = status === "active" ? "idle" : status;
                  agent.statusText = status === "active" ? "待命中" : status;
                }
              }
            );
          }

          // 首次连接成功后发送一条系统消息
          if (messages.value.length === 0) {
            messages.value.push({
              id: Date.now(),
              type: "system",
              content: "已成功连接到医疗AI多智能体咨询系统！",
              timestamp: new Date().toISOString(),
            });
          }
        }
      } catch (error) {
        console.error("健康检查失败:", error);
        isConnected.value = false;
        
        // 显示连接错误消息
        if (!messages.value.some(msg => msg.type === "error" && msg.content.includes("连接失败"))) {
          messages.value.push({
            id: Date.now(),
            type: "error",
            content: "连接到服务器失败，请检查服务是否正常运行。",
            timestamp: new Date().toISOString(),
          });
        }
      }
    };

    // 优化历史消息，只发送相关的用户消息和智能体回复
    const getRelevantHistory = () => {
      return messages.value
        .filter(
          (msg) =>
            msg.type === "user" ||
            msg.type === "agent" ||
            msg.type === "consensus"
        )
        .slice(-10) // 只保留最近10条相关消息
        .map((msg) => ({
          role: msg.type, // 使用role而不是type与后端模型兼容
          content: msg.content,
          agent: msg.agent,
          timestamp: msg.timestamp,
        }));
    };

// 发送消息 - 逐个智能体分析方式
    const sendMessageSequentialAgents = async (message) => {
      if (!message.trim() || !isConnected.value) return;

      isLoading.value = true;
      
      try {
        // 添加用户消息到界面
        messages.value.push({
          id: Date.now(),
          type: "user",
          content: message.trim(),
          timestamp: new Date().toISOString(),
        });
        
        // 滚动到底部
        await nextTick();
        scrollToBottom();

        // 获取相关历史记录
        const relevantHistory = getRelevantHistory();
        
        const requestData = {
          message: message,
          conversation_history: relevantHistory,
        };
        
        // 所有智能体类型
        const agentTypes = [
          { id: "oncologist", name: "肿瘤科医生" },
          { id: "radiologist", name: "影像科医生" },
          { id: "nurse", name: "肿瘤专科护士" },
          { id: "patient_advocate", name: "患者权益倡导者" },
          { id: "psychologist", name: "肿瘤心理专家" }
        ];
        
        // 存储所有智能体的响应结果
        const agentResponses = [];
        
        // 逐个请求智能体分析
        for (const agent of agentTypes) {
          try {
            // 更新智能体状态为思考中
            updateAgentStatus(agent.id, "thinking", "分析中...");
            
            console.log(`开始请求${agent.name}分析...`);
            
            // 调用单个智能体分析API
            const response = await apiClient.post(`/chat/agent/${agent.id}`, requestData);
            const agentData = response.data;
            
            console.log(`${agent.name}分析完成:`, agentData);
            
            // 更新智能体状态为已完成
            updateAgentStatus(agent.id, "completed", "已完成");
            
            // 添加智能体回复到消息列表
            messages.value.push({
              id: Date.now(),
              type: "agent",
              agent: agent.id,
              agentName: agent.name,
              content: agentData.content || "无内容",
              confidence: agentData.confidence || 0.5,
              recommendations: agentData.recommendations || [],
              timestamp: new Date().toISOString(),
            });
            
            // 保存响应用于后续生成共识
            agentResponses.push({
              agent: agent.id,
              agent_name: agent.name,
              content: agentData.content,
              confidence: agentData.confidence,
              recommendations: agentData.recommendations
            });
            
            // 每次添加消息后滚动到底部
            await nextTick();
            scrollToBottom();
            
          } catch (error) {
            console.error(`${agent.name}分析失败:`, error);
            
            // 更新智能体状态为错误
            updateAgentStatus(agent.id, "error", "分析失败");
            
            // 添加错误消息
            messages.value.push({
              id: Date.now(),
              type: "error",
              content: `${agent.name}分析失败，请稍后重试`,
              timestamp: new Date().toISOString(),
            });
          }
        }
        
        // 所有智能体分析完成后，生成共识
        if (agentResponses.length > 0) {
          try {
            // 添加共识分析消息
            messages.value.push({
              id: Date.now(),
              type: "thinking",
              content: "正在整合专家意见，生成最终共识...",
              timestamp: new Date().toISOString(),
            });
            
            await nextTick();
            scrollToBottom();
            
            // 可以调用生成共识的API，这里暂时使用前端处理
            // 如果需要后端处理，可以添加一个专门的共识生成API
            const consensus = generateConsensusFromResponses(agentResponses);
            
            // 更新推荐建议
            currentRecommendations.value = consensus.recommendations || [];
            
            // 添加共识消息
            messages.value.push({
              id: Date.now(),
              type: "consensus",
              content: consensus.content || "无法生成共识意见",
              confidence: consensus.confidence || 0.5,
              timestamp: new Date().toISOString(),
            });
            
          } catch (error) {
            console.error("生成共识失败:", error);
            messages.value.push({
              id: Date.now(),
              type: "error",
              content: "生成专家共识失败，请稍后重试",
              timestamp: new Date().toISOString(),
            });
          }
        }
      } finally {
        isLoading.value = false;
        
        // 重置智能体状态（延迟执行，让用户能看到完成状态）
        setTimeout(() => {
          agents.forEach((agent) => {
            updateAgentStatus(agent.id, "idle", "就绪");
          });
        }, 3000);
        
        // 确保滚动到底部
        await nextTick();
        scrollToBottom();
      }
    };
    
    // 从智能体响应生成共识（前端简化版本）
    const generateConsensusFromResponses = (agentResponses) => {
      // 计算平均置信度
      const confidence = agentResponses.reduce((sum, agent) => sum + (agent.confidence || 0), 0) / agentResponses.length;
      
      // 合并推荐建议，去重
      const allRecommendations = [];
      agentResponses.forEach(agent => {
        if (agent.recommendations && Array.isArray(agent.recommendations)) {
          allRecommendations.push(...agent.recommendations);
        }
      });
      
      // 去重
      const seen = new Set();
      const uniqueRecommendations = allRecommendations.filter(rec => {
        if (seen.has(rec)) return false;
        seen.add(rec);
        return true;
      });
      
      // 生成共识内容
      let content = "# 多智能体医疗团队共识意见\n\n";
      content += "基于我们多学科团队的综合分析：\n\n";
      
      // 添加各专家的核心观点
      agentResponses.forEach(agent => {
        if (agent.content) {
          const firstLine = agent.content.split('\n')[0] || '';
          content += `**${agent.agent_name}观点**：${firstLine}\n\n`;
        }
      });
      
      // 添加综合建议
      content += "## 综合建议\n";
      uniqueRecommendations.slice(0, 7).forEach((rec, index) => {
        content += `${index + 1}. ${rec}\n`;
      });
      
      content += "\n*请注意：以上建议仅供参考，具体治疗方案请遵循您的主治医生的建议。*";
      
      return {
        content,
        confidence,
        recommendations: uniqueRecommendations.slice(0, 7)
      };
    };

    // 新增：MDT讨论相关方法
    const switchView = (view) => {
      currentView.value = view;
    };

    const sendDiscussionMessage = async (message) => {
      if (!message.trim()) return;

      // 添加用户消息
      discussionMessages.value.push({
        id: Date.now(),
        type: 'user',
        content: message,
        timestamp: new Date()
      });

      isLoading.value = true;

      try {
        // 模拟MDT讨论流程
        await simulateMDTDiscussion(message);
      } catch (error) {
        console.error('讨论过程出错:', error);
        discussionMessages.value.push({
          id: Date.now(),
          type: 'system',
          messageType: 'error',
          title: '系统错误',
          content: '讨论过程中出现错误，请重试',
          timestamp: new Date()
        });
      } finally {
        isLoading.value = false;
        console.log('sendDiscussionMessage完成，isLoading设置为false，当前阶段:', discussionStage.value);
      }
    };

    const simulateMDTDiscussion = async (userMessage) => {
      // 阶段1：病例介绍和初始分析
      if (discussionStage.value === 'initial') {
        discussionMessages.value.push({
          id: Date.now(),
          type: 'system',
          messageType: 'stage',
          title: '开始MDT讨论',
          content: '专家团队正在分析患者病例...',
          timestamp: new Date()
        });

        updateDiscussionStage('analysis');
        
        // 模拟延迟
        await new Promise(resolve => setTimeout(resolve, 1500));

        // 各专家分析
        const experts = ['oncologist', 'nurse', 'psychologist'];
        for (const expert of experts) {
          await simulateExpertAnalysis(expert, userMessage);
          await new Promise(resolve => setTimeout(resolve, 1500));
        }

        // 完成初始分析阶段，等待用户进一步输入
        discussionMessages.value.push({
          id: Date.now(),
          type: 'system',
          messageType: 'info',
          title: '初始分析完成',
          content: '专家团队已完成初始分析。您可以继续提供更多信息，或者让我们进入下一阶段的团队讨论。',
          timestamp: new Date()
        });

        updateDiscussionStage('discussion');
        return;
      }
      
      // 阶段2：团队讨论
      if (discussionStage.value === 'discussion') {
        // 检查是否是"继续"命令
        if (userMessage && userMessage.trim().toLowerCase() === '继续') {
          discussionMessages.value.push({
            id: Date.now(),
            type: 'system',
            messageType: 'info',
            title: '进入团队讨论',
            content: '专家团队正在进行深入讨论...',
            timestamp: new Date()
          });
        } else if (userMessage && userMessage.trim()) {
          // 如果用户提供了额外信息，先处理
          discussionMessages.value.push({
            id: Date.now(),
            type: 'system',
            messageType: 'info',
            title: '补充信息已收到',
            content: '专家团队正在结合您提供的补充信息进行深入讨论...',
            timestamp: new Date()
          });
        }
        
        await simulateTeamDiscussion();
        
        discussionMessages.value.push({
          id: Date.now(),
          type: 'system',
          messageType: 'info',
          title: '团队讨论完成',
          content: '专家团队已完成讨论。您可以继续提问，或者输入"继续"形成专家共识。',
          timestamp: new Date()
        });
        
        updateDiscussionStage('consensus');
        return;
      }
      
      // 阶段3：形成共识
      if (discussionStage.value === 'consensus') {
        if (userMessage && userMessage.trim().toLowerCase() === '继续') {
          discussionMessages.value.push({
            id: Date.now(),
            type: 'system',
            messageType: 'info',
            title: '形成专家共识',
            content: '专家团队正在形成最终共识...',
            timestamp: new Date()
          });
        } else if (userMessage && userMessage.trim()) {
          discussionMessages.value.push({
            id: Date.now(),
            type: 'system',
            messageType: 'info',
            title: '意见已记录',
            content: '您的意见已被记录，专家团队正在形成最终共识...',
            timestamp: new Date()
          });
        }
        
        await simulateConsensusReached();
        
        discussionMessages.value.push({
          id: Date.now(),
          type: 'system',
          messageType: 'info',
          title: '专家共识已达成',
          content: '专家团队已达成共识。您可以继续询问详情，或者输入"继续"制定最终治疗方案。',
          timestamp: new Date()
        });
        
        updateDiscussionStage('plan');
        
        // 确保在共识完成后输入框保持可用
        console.log('共识阶段完成，当前阶段:', discussionStage.value, 'isLoading:', isLoading.value);
        
        // 强制触发界面更新
        await nextTick();
        return;
      }
      
      // 阶段4：制定最终方案
      if (discussionStage.value === 'plan') {
        if (userMessage && userMessage.trim().toLowerCase() === '继续') {
          discussionMessages.value.push({
            id: Date.now(),
            type: 'system',
            messageType: 'info',
            title: '制定治疗方案',
            content: '专家团队正在制定详细的治疗方案...',
            timestamp: new Date()
          });
        } else if (userMessage && userMessage.trim()) {
          discussionMessages.value.push({
            id: Date.now(),
            type: 'system',
            messageType: 'info',
            title: '需求已确认',
            content: '您的需求已确认，专家团队正在制定详细的治疗方案...',
            timestamp: new Date()
          });
        }
        
        await generateFinalPlan();
        
        discussionMessages.value.push({
          id: Date.now(),
          type: 'system',
          messageType: 'success',
          title: 'MDT讨论完成',
          content: '完整的治疗方案已制定完成。您可以继续就方案细节提问，或者对方案进行调整。',
          timestamp: new Date()
        });
        
        // 标记为已完成，但保持在plan阶段，允许继续对话
        updateDiscussionStage('completed');
        return;
      }
      
      // 如果已经完成所有阶段，处理后续对话
      if (discussionStage.value === 'completed' || finalPatientPlan.value) {
        discussionMessages.value.push({
          id: Date.now(),
          type: 'system',
          messageType: 'info',
          title: '专家回复',
          content: `关于您的问题："${userMessage}"，专家团队建议您参考已制定的治疗方案。如需调整方案或有其他疑问，请详细说明。`,
          timestamp: new Date()
        });
      }
    };

    const simulateExpertAnalysis = async (expertType, patientInfo) => {
      const expertData = {
        oncologist: {
          name: '李主任',
          title: '肿瘤科主任医师',
          avatar: '👨‍⚕️',
          analysis: '根据患者症状，建议进行进一步的影像学检查以明确诊断。',
          recommendations: ['CT扫描', '肿瘤标志物检测', '病理活检'],
          confidence: 85
        },
        nurse: {
          name: '王护士长',
          title: '肿瘤科护士长',
          avatar: '👩‍⚕️',
          analysis: '患者需要心理支持和营养指导，建议制定个性化护理计划。',
          recommendations: ['心理疏导', '营养评估', '生活质量评估'],
          confidence: 90
        },
        psychologist: {
          name: '张医生',
          title: '心理医生',
          avatar: '🧠',
          analysis: '患者可能存在焦虑情绪，需要心理干预和家属支持。',
          recommendations: ['心理评估', '认知行为治疗', '家庭支持'],
          confidence: 88
        }
      };

      const expert = expertData[expertType];
      
      discussionMessages.value.push({
        id: Date.now(),
        type: 'role',
        role: expertType,
        expert: expert,
        content: {
          mainPoint: expert.analysis,
          analysis: `基于我的专业经验，${expert.analysis}`,
          recommendations: expert.recommendations,
          risks: ['需要密切观察', '定期随访'],
          evidence: '基于临床指南和最佳实践',
          metrics: { confidence: expert.confidence }
        },
        timestamp: new Date()
      });
    };

    const simulateTeamDiscussion = async () => {
      discussionMessages.value.push({
        id: Date.now(),
        type: 'system',
        messageType: 'info',
        title: '团队讨论',
        content: '专家团队正在就诊疗方案进行深入讨论...',
        timestamp: new Date()
      });

      await new Promise(resolve => setTimeout(resolve, 2000));
    };

    const simulateConsensusReached = async () => {
      const consensus = {
        score: 92,
        conclusion: '经过充分讨论，专家团队就患者的诊疗方案达成高度共识',
        treatment: '综合治疗方案',
        opinions: [
          { expert: '肿瘤科', opinion: '建议手术治疗结合化疗', agreement: 95 },
          { expert: '护理科', opinion: '制定全程护理计划', agreement: 98 },
          { expert: '心理科', opinion: '提供心理支持服务', agreement: 90 }
        ],
        considerations: ['患者年龄', '身体状况', '家庭支持'],
        risks: ['手术风险', '化疗副作用'],
        benefits: ['提高生存率', '改善生活质量'],
        nextSteps: ['完善术前检查', '制定详细方案', '患者教育']
      };

      consensusStats.value = {
        totalExperts: 3,
        agreementRate: 94,
        consensusTime: '15分钟',
        discussionRounds: 2
      };

      discussionMessages.value.push({
        id: Date.now(),
        type: 'consensus',
        consensus: consensus,
        timestamp: new Date()
      });
    };

    const generateFinalPlan = async () => {
      await new Promise(resolve => setTimeout(resolve, 1500));

      finalPatientPlan.value = {
        patient: {
          name: '患者姓名',
          age: '45岁',
          gender: '女性',
          id: 'P2024001',
          diagnosis: '疑似恶性肿瘤'
        },
        treatment: {
          primary: '手术治疗',
          secondary: ['化疗', '放疗'],
          duration: '6个月',
          location: '肿瘤科病房'
        },
        medications: [
          { name: '化疗药物A', dosage: '100mg', frequency: '每周一次', duration: '6周' },
          { name: '止痛药', dosage: '50mg', frequency: '必要时', duration: '长期' }
        ],
        timeline: [
          { phase: '术前准备', duration: '1周', activities: ['完善检查', '术前评估'] },
          { phase: '手术治疗', duration: '1天', activities: ['手术', '术后监护'] },
          { phase: '术后恢复', duration: '2周', activities: ['伤口护理', '康复训练'] },
          { phase: '辅助治疗', duration: '6个月', activities: ['化疗', '定期复查'] }
        ],
        outcomes: {
          expected: ['肿瘤完全切除', '症状缓解', '生活质量改善'],
          risks: ['手术并发症', '化疗副作用', '复发风险']
        },
        followUp: {
          schedule: '术后1周、1个月、3个月、6个月',
          tests: ['血常规', '肿瘤标志物', 'CT复查'],
          contact: '24小时急诊热线：400-123-4567'
        },
        team: [
          { name: '李主任', role: '主治医师', contact: 'li@hospital.com' },
          { name: '王护士长', role: '责任护士', contact: 'wang@hospital.com' },
          { name: '张医生', role: '心理医生', contact: 'zhang@hospital.com' }
        ],
        alternatives: [
          { option: '保守治疗', description: '药物治疗为主' },
          { option: '姑息治疗', description: '缓解症状，提高生活质量' }
        ]
      };

      discussionMessages.value.push({
        id: Date.now(),
        type: 'system',
        messageType: 'success',
        title: '治疗方案已生成',
        content: '专家团队已为患者制定了详细的治疗方案',
        timestamp: new Date()
      });

      // 完成所有阶段
      discussionStages.forEach(stage => stage.completed = true);
    };

    const updateDiscussionStage = (newStage) => {
      console.log('更新讨论阶段:', discussionStage.value, '->', newStage);
      
      const currentIndex = discussionStages.findIndex(s => s.id === discussionStage.value);
      if (currentIndex >= 0) {
        discussionStages[currentIndex].completed = true;
      }
      discussionStage.value = newStage;
      
      // 强制更新UI
      nextTick(() => {
        console.log('阶段更新完成，当前阶段:', discussionStage.value);
        console.log('当前placeholder:', discussionPlaceholder.value);
      });
    };

    // PatientPlan组件的事件处理函数
    const approvePlan = () => {
      console.log('用户批准了治疗方案');
      // 可以在这里添加批准方案的逻辑
      discussionMessages.value.push({
        id: Date.now(),
        type: "system",
        content: "✅ 您已批准此治疗方案。方案将被记录并可供后续参考。",
        timestamp: new Date().toISOString(),
      });
    };

    const requestModification = (modification) => {
      console.log('用户请求修改方案:', modification);
      // 可以在这里添加请求修改的逻辑
      discussionMessages.value.push({
        id: Date.now(),
        type: "system", 
        content: `📝 您的修改请求已记录：${modification || '请在输入框中详细说明您希望的修改内容'}`,
        timestamp: new Date().toISOString(),
      });
    };

    const exportPlan = () => {
      console.log('导出治疗方案');
      // 可以在这里添加导出方案的逻辑
      if (finalPatientPlan.value) {
        const planData = JSON.stringify(finalPatientPlan.value, null, 2);
        const blob = new Blob([planData], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `治疗方案_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        discussionMessages.value.push({
          id: Date.now(),
          type: "system",
          content: "📄 治疗方案已导出为JSON文件。",
          timestamp: new Date().toISOString(),
        });
      }
    };

    const printPlan = () => {
      console.log('打印治疗方案');
      // 可以在这里添加打印方案的逻辑
      if (finalPatientPlan.value) {
        const printWindow = window.open('', '_blank');
        printWindow.document.write(`
          <html>
            <head>
              <title>治疗方案</title>
              <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                .section { margin: 20px 0; }
                .section h3 { color: #666; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
              </style>
            </head>
            <body>
              <h1>MDT治疗方案</h1>
              <div class="section">
                <h3>方案内容</h3>
                <pre>${JSON.stringify(finalPatientPlan.value, null, 2)}</pre>
              </div>
              <div class="section">
                <h3>生成时间</h3>
                <p>${new Date().toLocaleString()}</p>
              </div>
            </body>
          </html>
        `);
        printWindow.document.close();
        printWindow.print();
        
        discussionMessages.value.push({
          id: Date.now(),
          type: "system",
          content: "🖨️ 治疗方案打印窗口已打开。",
          timestamp: new Date().toISOString(),
        });
      }
    };
    
    // 发送消息主函数 - 异步处理方式
    const sendMessage = async (message) => {
      if (!message.trim() || !isConnected.value) return;

      isLoading.value = true;
      
      try {
        // 添加用户消息到界面
        messages.value.push({
          id: Date.now(),
          type: "user",
          content: message.trim(),
          timestamp: new Date().toISOString(),
        });
        
        // 滚动到底部
        await nextTick();
        scrollToBottom();

        // 获取相关历史记录
        const relevantHistory = getRelevantHistory();
        
        const requestData = {
          message: message,
          conversation_history: relevantHistory,
        };
        
        // 发送异步请求
        console.log('发送异步聊天请求...');
        const asyncResponse = await apiClient.post('/chat/async', requestData);
        const { request_id, initial_response } = asyncResponse.data;
        
        console.log('收到异步响应，请求ID:', request_id);
        
        // 立即显示初步响应
        const initialMessageId = Date.now();
        messages.value.push({
          id: initialMessageId,
          type: "async_initial",
          content: initial_response,
          request_id: request_id,
          timestamp: new Date().toISOString(),
        });
        
        await nextTick();
        scrollToBottom();
        
        // 开始轮询获取进度和结果
        await pollChatStatus(request_id, initialMessageId);
        
      } catch (error) {
        console.error('异步请求失败:', error);
        
        // 添加错误消息
        messages.value.push({
          id: Date.now(),
          type: "error",
          content: error.response?.data?.detail || "发送请求失败，请稍后重试",
          timestamp: new Date().toISOString(),
        });
        
      } finally {
        isLoading.value = false;
        
        // 确保滚动到底部
        await nextTick();
        scrollToBottom();
      }
    };
    
    // 轮询聊天状态
    const pollChatStatus = async (requestId, initialMessageId) => {
      let pollInterval;
      let processedAgents = new Set(); // 跟踪已处理的智能体
      
      try {
        // 轮询函数
        const poll = async () => {
          try {
            const statusResponse = await apiClient.get(`/chat/status/${requestId}`);
            const { status, agent_responses, consensus } = statusResponse.data;
            
            console.log('轮询状态:', status, '已获取响应的智能体:', Object.keys(agent_responses).length);
            
            // 更新初步响应消息的状态
            const initialMessage = messages.value.find(msg => msg.id === initialMessageId);
            if (initialMessage) {
              initialMessage.status = status;
            }
            
            // 所有智能体类型及其显示名称
            const agentTypes = {
              "oncologist": "肿瘤科医生",
              "radiologist": "影像科医生",
              "nurse": "肿瘤专科护士",
              "patient_advocate": "患者权益倡导者",
              "psychologist": "肿瘤心理专家"
            };
            
            // 检查并显示新的智能体响应
            for (const [agentId, agentData] of Object.entries(agent_responses)) {
              if (!processedAgents.has(agentId)) {
                processedAgents.add(agentId);
                const agentName = agentTypes[agentId] || agentId;
                
                // 更新智能体状态
                updateAgentStatus(agentId, "completed", "分析完成");
                
                // 添加智能体回复
                messages.value.push({
                  id: Date.now(),
                  type: "agent",
                  agent: agentId,
                  agentName: agentName,
                  content: agentData.content || "无内容",
                  confidence: agentData.confidence || 0.5,
                  recommendations: agentData.recommendations || [],
                  timestamp: new Date().toISOString(),
                });
                
                await nextTick();
                scrollToBottom();
              }
            }
            
            // 如果处理完成且有共识结果
            if (status === "completed" && consensus) {
              // 更新推荐建议
              currentRecommendations.value = consensus.recommendations || [];
              
              // 添加共识消息
              messages.value.push({
                id: Date.now(),
                type: "consensus",
                content: consensus.content || "无法生成共识意见",
                confidence: consensus.confidence || 0.5,
                timestamp: new Date().toISOString(),
              });
              
              await nextTick();
              scrollToBottom();
              
              // 清除轮询
              clearInterval(pollInterval);
              
              // 重置智能体状态（延迟执行）
              setTimeout(() => {
                agents.forEach((agent) => {
                  updateAgentStatus(agent.id, "idle", "就绪");
                });
              }, 3000);
            }
            // 如果发生错误
            else if (status === "error") {
              messages.value.push({
                id: Date.now(),
                type: "error",
                content: "处理分析请求时发生错误，请稍后重试",
                timestamp: new Date().toISOString(),
              });
              
              clearInterval(pollInterval);
            }
            
          } catch (error) {
            console.error('轮询状态失败:', error);
            // 继续轮询，直到达到最大次数
          }
        };
        
        // 立即执行一次轮询
        await poll();
        
        // 设置轮询间隔（1.5秒轮询一次）
        pollInterval = setInterval(poll, 1500);
        
        // 设置最大轮询时间（5分钟）
        setTimeout(() => {
          if (pollInterval) {
            clearInterval(pollInterval);
            
            // 检查是否已完成
            const initialMessage = messages.value.find(msg => msg.id === initialMessageId);
            if (initialMessage && initialMessage.status !== "completed") {
              messages.value.push({
                id: Date.now(),
                type: "warning",
                content: "分析超时，请刷新页面后查看结果或重新提交请求",
                timestamp: new Date().toISOString(),
              });
              
              // 重置智能体状态
              agents.forEach((agent) => {
                updateAgentStatus(agent.id, "idle", "就绪");
              });
            }
          }
        }, 300000); // 5分钟 = 300000毫秒
        
      } catch (error) {
        console.error('轮询过程中发生错误:', error);
        if (pollInterval) {
          clearInterval(pollInterval);
        }
      }
    };

    const updateAgentStatus = (agentId, status, statusText) => {
      const agent = agents.find((a) => a.id === agentId);
      if (agent) {
        agent.status = status;
        agent.statusText = statusText;
      }
    };

    const scrollToBottom = () => {
      // 使用requestAnimationFrame确保在DOM更新后执行滚动
      requestAnimationFrame(() => {
        const container = document.querySelector(".messages-container");
        if (container) {
          container.scrollTop = container.scrollHeight;
        }
      });
    };

    onMounted(() => {
      // 立即进行一次健康检查
      checkHealth();

      // 设置定期健康检查（每10秒，减少频率避免频繁请求）
      healthCheckInterval.value = setInterval(checkHealth, 100000);
    });

    // 讨论阶段占位符文本已在上方定义

    onUnmounted(() => {
      // 清除健康检查定时器
      if (healthCheckInterval.value) {
        clearInterval(healthCheckInterval.value);
      }
    });

    return {
      isConnected,
      isLoading,
      messages,
      currentRecommendations,
      agents,
      inputPlaceholder,
      sendMessage,
      scrollToBottom,
      // MDT讨论相关
      currentView,
      discussionMessages,
      discussionStage,
      discussionStages,
      finalPatientPlan,
      consensusStats,
      discussionPlaceholder,
      sendDiscussionMessage,
      switchView,
      updateDiscussionStage,
      // PatientPlan事件处理函数
      approvePlan,
      requestModification,
      exportPlan,
      printPlan
    };
  },
};
</script>

<style scoped>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

.app-container {
  width: 100vw;
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    "Helvetica Neue", Arial, sans-serif;
}

/* 视图切换样式 */
.view-toggle {
  display: flex;
  background: rgba(255, 255, 255, 0.15);
  border-radius: 30px;
  padding: 6px;
  margin: 0 20px;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.toggle-btn {
  flex: 1;
  padding: 12px 20px;
  border: none;
  border-radius: 24px;
  background: transparent;
  color: white;
  cursor: pointer;
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  font-size: 14px;
  font-weight: 600;
  position: relative;
  overflow: hidden;
}

.toggle-btn::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.3), rgba(255, 255, 255, 0.1));
  opacity: 0;
  transition: opacity 0.3s ease;
  border-radius: 24px;
}

.toggle-btn.active {
  background: rgba(255, 255, 255, 0.25);
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
  transform: translateY(-1px);
}

.toggle-btn.active::before {
  opacity: 1;
}

.toggle-btn:hover:not(.active) {
  background: rgba(255, 255, 255, 0.15);
  transform: translateY(-1px);
}

/* MDT讨论界面样式 */
.discussion-view {
  display: flex;
  flex: 1;
  gap: 24px;
  padding: 24px;
  overflow: hidden;
}

.discussion-main {
  flex: 2;
  display: flex;
  flex-direction: column;
  background: rgba(255, 255, 255, 0.98);
  border-radius: 20px;
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.12);
  overflow: visible;
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.3);
}

.progress-panel {
  background: rgba(255, 255, 255, 0.98);
  border-radius: 20px;
  padding: 24px;
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.12);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.3);
  min-width: 320px;
}

.progress-stages {
  margin-bottom: 20px;
}

.stage-item {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 16px 0;
  border-bottom: 1px solid rgba(240, 240, 240, 0.6);
  transition: all 0.3s ease;
  position: relative;
}

.stage-item:last-child {
  border-bottom: none;
}

.stage-item:hover {
  background: rgba(59, 130, 246, 0.05);
  border-radius: 12px;
  padding-left: 12px;
  padding-right: 12px;
}

.stage-number {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  font-weight: 700;
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  background: linear-gradient(135deg, #f1f3f4, #e8eaed);
  color: #9aa0a6;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.stage-item.completed .stage-number {
  background: linear-gradient(135deg, #34d399, #10b981);
  color: white;
  box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
  transform: scale(1.05);
}

.stage-item.active .stage-number {
  background: linear-gradient(135deg, #60a5fa, #3b82f6);
  color: white;
  animation: pulse 2s infinite;
  box-shadow: 0 4px 20px rgba(59, 130, 246, 0.5);
  transform: scale(1.1);
}

@keyframes pulse {
  0% { transform: scale(1); }
  50% { transform: scale(1.1); }
  100% { transform: scale(1); }
}

.stage-info {
  flex: 1;
}

.stage-name {
  font-weight: 600;
  color: #2c3e50;
  margin-bottom: 2px;
}

.stage-description {
  font-size: 12px;
  color: #7f8c8d;
}

.consensus-stats {
  margin-top: 24px;
  padding-top: 24px;
  border-top: 1px solid rgba(240, 240, 240, 0.6);
}

.consensus-stats h4 {
  font-size: 18px;
  font-weight: 700;
  color: #1f2937;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.consensus-stats h4::before {
  content: '📊';
  font-size: 20px;
}

.stats-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

.stat-item {
  text-align: center;
  padding: 20px 16px;
  background: linear-gradient(135deg, #f8fafc, #f1f5f9);
  border-radius: 16px;
  border: 1px solid rgba(226, 232, 240, 0.8);
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.stat-item::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: linear-gradient(90deg, #3b82f6, #8b5cf6);
  opacity: 0;
  transition: opacity 0.3s ease;
}

.stat-item:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
}

.stat-item:hover::before {
  opacity: 1;
}

.stat-value {
  display: block;
  font-size: 28px;
  font-weight: 800;
  background: linear-gradient(135deg, #3b82f6, #8b5cf6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 8px;
  line-height: 1;
}

.stat-label {
  font-size: 12px;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 1px;
  font-weight: 600;
}

/* 传统聊天界面样式 */
.chat-view {
  display: flex;
  flex: 1;
  gap: 24px;
  padding: 24px;
}

.chat-container {
  flex: 2;
  display: flex;
  flex-direction: column;
  background: rgba(255, 255, 255, 0.98);
  border-radius: 20px;
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.12);
  overflow: hidden;
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.3);
}

.app-header {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  padding: 1rem 2rem;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  max-width: 1400px;
  margin: 0 auto;
}

.header-controls {
  display: flex;
  align-items: center;
  gap: 20px;
}

.app-header h1 {
  color: #2c3e50;
  font-size: 1.5rem;
  font-weight: 600;
}

.header-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  color: #666;
  font-size: 0.9rem;
}

.status-indicator {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #e74c3c;
  transition: background 0.3s;
}

.status-indicator.connected {
  background: #2ecc71;
  box-shadow: 0 0 10px rgba(46, 204, 113, 0.5);
}

.main-content {
  flex: 1;
  overflow: hidden;
}

.agents-panel,
.recommendations-panel {
  background: rgba(255, 255, 255, 0.98);
  backdrop-filter: blur(20px);
  border-radius: 20px;
  padding: 24px;
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.12);
  overflow-y: auto;
  border: 1px solid rgba(255, 255, 255, 0.3);
  min-width: 320px;
}

.agents-panel h3,
.recommendations-panel h3 {
  color: #1f2937;
  margin-bottom: 20px;
  font-size: 18px;
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 8px;
}

.agents-panel h3::before {
  content: '👥';
  font-size: 20px;
}

.recommendations-panel h3::before {
  content: '💡';
  font-size: 20px;
}

.agent-card {
  background: #f8f9fa;
  border-radius: 10px;
  padding: 1rem;
  margin-bottom: 1rem;
  display: flex;
  gap: 1rem;
  transition: transform 0.2s, box-shadow 0.2s;
}

.agent-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.agent-avatar {
  font-size: 2rem;
  width: 50px;
  height: 50px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: white;
  border-radius: 10px;
}

.agent-info {
  flex: 1;
}

.agent-info h4 {
  color: #2c3e50;
  font-size: 1rem;
  margin-bottom: 0.25rem;
}

.agent-specialty {
  color: #7f8c8d;
  font-size: 0.85rem;
  margin-bottom: 0.5rem;
}

.agent-status {
  display: inline-block;
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 500;
}

.agent-status.idle {
  background: #ecf0f1;
  color: #7f8c8d;
}

.agent-status.working {
  background: #fff3cd;
  color: #856404;
  animation: pulse 1.5s infinite;
}

.agent-status.completed {
  background: #d4edda;
  color: #155724;
}

@keyframes pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.5;
  }
}

.chat-container {
  display: flex;
  flex-direction: column;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
  overflow: hidden;
}

.recommendations-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.recommendation-item {
  display: flex;
  gap: 0.75rem;
  padding: 0.75rem;
  background: #f8f9fa;
  border-radius: 8px;
  border-left: 3px solid #667eea;
  transition: transform 0.2s;
}

.recommendation-item:hover {
  transform: translateX(5px);
}

.rec-number {
  flex-shrink: 0;
  width: 24px;
  height: 24px;
  background: #667eea;
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.85rem;
  font-weight: bold;
}

.rec-text {
  color: #2c3e50;
  font-size: 0.9rem;
  line-height: 1.5;
}

.no-recommendations {
  text-align: center;
  padding: 2rem 1rem;
  color: #7f8c8d;
}

.no-recommendations p {
  margin-bottom: 0.5rem;
}

.hint {
  font-size: 0.85rem;
  color: #95a5a6;
}

/* 快速启动面板样式 */
.quick-start-panel {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 400px;
  padding: 40px 20px;
}

.welcome-message {
  text-align: center;
  max-width: 800px;
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 250, 252, 0.95));
  border-radius: 24px;
  padding: 40px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.3);
}

.welcome-message h2 {
  color: #1f2937;
  font-size: 28px;
  font-weight: 700;
  margin-bottom: 16px;
  background: linear-gradient(135deg, #3b82f6, #8b5cf6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.welcome-message p {
  color: #64748b;
  font-size: 16px;
  margin-bottom: 24px;
  line-height: 1.6;
}

.flow-preview {
  background: linear-gradient(135deg, #f1f5f9, #e2e8f0);
  border-radius: 16px;
  padding: 20px;
  margin: 24px 0;
  font-size: 14px;
  font-weight: 600;
  color: #475569;
  border: 1px solid rgba(226, 232, 240, 0.8);
  line-height: 1.8;
}

.start-discussion-btn {
  background: linear-gradient(135deg, #3b82f6, #8b5cf6);
  color: white;
  border: none;
  border-radius: 16px;
  padding: 16px 32px;
  font-size: 16px;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 8px 25px rgba(59, 130, 246, 0.3);
  position: relative;
  overflow: hidden;
}

.start-discussion-btn::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
  transition: left 0.5s ease;
}

.start-discussion-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 35px rgba(59, 130, 246, 0.4);
}

.start-discussion-btn:hover::before {
  left: 100%;
}

.start-discussion-btn:active {
  transform: translateY(0);
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .discussion-view {
    gap: 16px;
  }
  
  .agents-panel,
  .recommendations-panel {
    min-width: 280px;
  }
}

@media (max-width: 992px) {
  .chat-view {
    gap: 16px;
    padding: 16px;
  }
  
  .discussion-view {
    padding: 16px;
  }
  
  .stats-grid {
    grid-template-columns: 1fr;
    gap: 12px;
  }
}

@media (max-width: 768px) {
  .app-header {
    padding: 16px;
  }
  
  .view-toggle {
    flex-direction: column;
    gap: 8px;
    width: 100%;
  }
  
  .view-toggle button {
    width: 100%;
    justify-content: center;
  }
  
  .discussion-view {
    flex-direction: column;
    padding: 16px;
    gap: 16px;
  }
  
  .progress-panel {
    order: -1;
    margin-bottom: 0;
  }
  
  .chat-view {
    flex-direction: column;
    padding: 16px;
    gap: 16px;
  }
  
  .agents-panel,
  .recommendations-panel {
    margin-top: 0;
    min-width: auto;
    width: 100%;
  }
  
  .stage-item {
    padding: 12px 0;
  }
  
  .stage-item:hover {
    padding-left: 8px;
    padding-right: 8px;
  }
  
  .stat-item {
    padding: 16px 12px;
  }
}

@media (max-width: 480px) {
  .app-header h1 {
    font-size: 20px;
  }
  
  .view-toggle button {
    padding: 10px 16px;
    font-size: 14px;
  }
  
  .discussion-view,
  .chat-view {
    padding: 12px;
  }
  
  .agents-panel,
  .recommendations-panel {
    padding: 16px;
  }
  
  .stage-number {
    width: 28px;
    height: 28px;
    font-size: 12px;
  }
  
  .stat-value {
    font-size: 24px;
  }
  
  .consensus-stats h4 {
    font-size: 16px;
  }
}
</style>
