
import React, { useState, useCallback, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { DataUpload } from './components/DataUpload';
import { TrainingMonitor } from './components/TrainingMonitor';
import { EvaluationResults } from './components/EvaluationResults';
import { ClusteringAnalysis } from './components/ClusteringAnalysis';
import { ImageInspector } from './components/ImageInspector';
import { ChatIA } from './components/ChatIA';
import { ExplanationHub } from './components/ExplanationHub';
import { DEFAULT_CONFIG, AppSection, ALL_EXPLANATIONS, SIDEBAR_EXPLANATIONS } from './constants';
import type { TrainingMetrics, ClassificationReportData, ConfusionMatrixData, ClusterVisualizationData, IndividualEvaluationData, ExplanationTopic, TrainingStatus, ConfigDataEntry, SidebarConfig, SampleImage, ErrorAnalysisItem, ROCCurveData, ClusterPoint, ChatMessage, UserOrModelMessage, SystemLogMessage } from './types';
import { generateMockMetrics, generateMockClassificationReport, generateMockConfusionMatrix, generateMockClusterData, generateMockErrorAnalysis, generateMockCAMImage, generateMockROCCurveData, generateMockAugmentedEmbeddings, generateMockPRCurveData, generateMockUncertaintyScore } from './services/mockDataService';
import { exportResultsToCSV, exportTrainingMetricsToCSV } from './services/csvExporter'; 
import JSZip from 'jszip';
import { GoogleGenAI, Chat } from "@google/genai";

const App: React.FC = () => {
  const [config, setConfig] = useState<SidebarConfig>(DEFAULT_CONFIG);
  const [currentSection, setCurrentSection] = useState<AppSection>(AppSection.DATA_CONFIG);
  const [zipFile, setZipFile] = useState<File | null>(null);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetrics | null>(null);
  const [evaluationReport, setEvaluationReport] = useState<ClassificationReportData | null>(null);
  const [confusionMatrix, setConfusionMatrix] = useState<ConfusionMatrixData | null>(null);
  const [errorAnalysisData, setErrorAnalysisData] = useState<ErrorAnalysisItem[] | null>(null);
  const [clusterData, setClusterData] = useState<ClusterVisualizationData | null>(null);
  const [augmentedEmbeddingsData, setAugmentedEmbeddingsData] = useState<ClusterPoint[] | null>(null);
  const [individualEval, setIndividualEval] = useState<IndividualEvaluationData | null>(null);
  const [camImage, setCamImage] = useState<string | null>(null);
  const [uploadedImageForEval, setUploadedImageForEval] = useState<File | null>(null);
  const [rocCurveData, setRocCurveData] = useState<ROCCurveData | null>(null);
  const [prCurveData, setPrCurveData] = useState<ROCCurveData | null>(null); 
  
  const [numClassesInData, setNumClassesInData] = useState<number>(DEFAULT_CONFIG.numClasses);
  const [classNamesInData, setClassNamesInData] = useState<string[]>(Array.from({ length: DEFAULT_CONFIG.numClasses }, (_, i) => `Classe ${String.fromCharCode(65 + i)}`));
  const [sampleImagesFromZip, setSampleImagesFromZip] = useState<SampleImage[]>([]);
  const [isProcessingZip, setIsProcessingZip] = useState<boolean>(false);
  const [areResultsAvailable, setAreResultsAvailable] = useState<boolean>(false); 
  const [isTrainingComplete, setIsTrainingComplete] = useState<boolean>(false);
  const [trainingLog, setTrainingLog] = useState<string[]>([]);

  // ChatIA State
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState<string>('');
  const [isChatLoading, setIsChatLoading] = useState<boolean>(false);
  const [geminiChatInstance, setGeminiChatInstance] = useState<Chat | null>(null);
  const [userClassificationType, setUserClassificationType] = useState<string | null>(null);
  const [awaitingClassificationType, setAwaitingClassificationType] = useState<boolean>(false);
  const [simulatedAgentLog, setSimulatedAgentLog] = useState<string[]>([]);


  const handleConfigChange = useCallback((newConfig: SidebarConfig) => {
    if (newConfig.fineTune && zipFile && newConfig.numClasses !== numClassesInData) {
      setConfig({ ...newConfig, numClasses: numClassesInData });
    } else {
      setConfig(newConfig);
    }
  }, [zipFile, numClassesInData]);


  const handleZipUpload = async (file: File) => {
    setZipFile(file);
    setIsProcessingZip(true);
    setSampleImagesFromZip([]); 
    setAreResultsAvailable(false);
    setIsTrainingComplete(false);
    setTrainingLog([]);
    setEvaluationReport(null);
    setConfusionMatrix(null);
    setErrorAnalysisData(null);
    setClusterData(null);
    setAugmentedEmbeddingsData(null);
    setRocCurveData(null);
    setPrCurveData(null);
    setTrainingMetrics(null);
    setGeminiChatInstance(null); 
    setChatMessages([]);
    setUserClassificationType(null);
    setAwaitingClassificationType(false);
    setSimulatedAgentLog([]);


    try {
      const zip = await JSZip.loadAsync(file);
      const classFolders: { [key: string]: JSZip.JSZipObject[] } = {};
      const imageFileExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp'];
      
      zip.forEach((relativePath, zipEntry) => {
        if (!zipEntry.dir && relativePath.includes('/')) {
          const parts = relativePath.split('/');
          const topLevelFolder = parts[0];
          
          if (parts.length > 1 && !zipEntry.name.startsWith('__MACOSX')) { 
             if (parts.length === 2 || (parts.length > 2 && parts[1] !== '' && imageFileExtensions.some(ext => parts[parts.length -1].toLowerCase().endsWith(ext)))) {
                 if (!classFolders[topLevelFolder]) {
                    classFolders[topLevelFolder] = [];
                 }
                 if (imageFileExtensions.some(ext => zipEntry.name.toLowerCase().endsWith(ext))) {
                    classFolders[topLevelFolder].push(zipEntry);
                 }
             }
          }
        }
      });

      const detectedClassNames = Object.keys(classFolders).filter(name => classFolders[name].length > 0);
      const detectedNumClasses = detectedClassNames.length;

      if (detectedNumClasses === 0) {
        alert("Nenhuma pasta de classe com imagens válidas encontrada no arquivo ZIP. Certifique-se de que o ZIP contenha pastas de primeiro nível, cada uma representando uma classe com arquivos de imagem (jpg, png, gif, bmp) dentro.");
        setNumClassesInData(DEFAULT_CONFIG.numClasses);
        setClassNamesInData(Array.from({ length: DEFAULT_CONFIG.numClasses }, (_, i) => `Classe Padrão ${String.fromCharCode(65 + i)}`));
        setIsProcessingZip(false);
        return;
      }
      
      setNumClassesInData(detectedNumClasses);
      setClassNamesInData(detectedClassNames);

      const extractedSamples: SampleImage[] = [];
      const maxSamplesPerClass = 3; 
      const totalMaxSamples = 10; 

      for (const className of detectedClassNames) {
        if (extractedSamples.length >= totalMaxSamples) break;
        const imageFiles = classFolders[className];
        if (imageFiles && imageFiles.length > 0) {
          for (let i = 0; i < Math.min(imageFiles.length, maxSamplesPerClass); i++) {
            if (extractedSamples.length >= totalMaxSamples) break;
            try {
                const imageFile = imageFiles[i];
                const blob = await imageFile.async('blob');
                const dataUrl = await new Promise<string>((resolve, reject) => {
                    const reader = new FileReader();
                    reader.onloadend = () => resolve(reader.result as string);
                    reader.onerror = reject;
                    reader.readAsDataURL(blob);
                });
                extractedSamples.push({ className, imageDataUrl: dataUrl, fileName: imageFile.name.split('/').pop() || imageFile.name });
            } catch (imgError) {
                console.error(`Erro ao processar imagem ${imageFiles[i].name} da classe ${className}:`, imgError);
            }
          }
        }
      }
      setSampleImagesFromZip(extractedSamples);

      if (config.fineTune) {
        setConfig(prev => ({ ...prev, numClasses: detectedNumClasses }));
      } else {
        if (config.numClasses === DEFAULT_CONFIG.numClasses || config.numClasses === 0 || config.numClasses === 2) {
           setConfig(prev => ({ ...prev, numClasses: detectedNumClasses }));
        }
      }

    } catch (error) {
      console.error("Erro ao processar arquivo ZIP:", error);
      alert("Ocorreu um erro ao processar o arquivo ZIP. Verifique se é um arquivo ZIP válido e tente novamente.");
      setNumClassesInData(DEFAULT_CONFIG.numClasses); 
      setClassNamesInData(Array.from({ length: DEFAULT_CONFIG.numClasses }, (_, i) => `Classe Padrão ${String.fromCharCode(65 + i)}`));
    } finally {
      setIsProcessingZip(false);
    }
    setCurrentSection(AppSection.DATA_CONFIG);
  };
  
  const saveConfigToJson = () => {
    const effectiveNumClasses = config.fineTune && zipFile ? numClassesInData : config.numClasses;
    const configToSave: ConfigDataEntry[] = [
        { parameter: "Modelo", value: config.modelName },
        { parameter: "Fine-Tuning Completo", value: config.fineTune ? "Sim" : "Não" },
        { parameter: "Número de Classes Efetivo", value: effectiveNumClasses.toString() },
        { parameter: "Épocas", value: config.epochs.toString() },
        { parameter: "Taxa de Aprendizagem", value: config.learningRate.toString() },
        { parameter: "Tamanho de Lote", value: config.batchSize.toString() },
        { parameter: "Divisão Treino", value: config.trainSplit.toString() },
        { parameter: "Divisão Validação", value: config.validSplit.toString() },
        { parameter: "Estratégia de Validação", value: config.validationStrategy },
        { parameter: "Regularização L2", value: config.l2Lambda.toString() },
        { parameter: "Paciência Early Stopping", value: config.patience.toString() },
        { parameter: "Usar Perda Ponderada", value: config.useWeightedLoss ? "Sim" : "Não" },
        { parameter: "Apresentar Score de Incerteza", value: config.simulatedUncertainty ? "Sim" : "Não" },
        { parameter: "Otimizador", value: config.optimizerName },
        { parameter: "Agendador LR", value: config.lrSchedulerName },
        { parameter: "Aumento de Dados", value: config.dataAugmentationMethod },
        { parameter: "Método XAI", value: config.camMethod },
    ];
    const jsonString = JSON.stringify(configToSave, null, 2);
    const blob = new Blob([jsonString], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `config_${config.modelName}_execucao_${new Date().toISOString().slice(0,10)}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleExportResults = () => {
    const currentClassNames = config.fineTune && zipFile ? classNamesInData : Array.from({ length: config.numClasses }, (_, i) => `Classe ${String.fromCharCode(65 + i)}`);
    exportResultsToCSV(
        trainingMetrics,
        evaluationReport,
        confusionMatrix,
        errorAnalysisData,
        clusterData,
        individualEval,
        currentClassNames,
        config.modelName,
        rocCurveData, 
        prCurveData   
    );
  };

  const handleExportTrainingMetrics = () => {
    exportTrainingMetricsToCSV(trainingMetrics, config.modelName);
  };

  const getResultsAsTextContext = useCallback((): string => {
    let context = "## Contexto dos Resultados do Modelo de Classificação de Imagens ##\n\n";
    const effectiveClassNames = config.fineTune && zipFile && classNamesInData.length > 0 ? classNamesInData : Array.from({ length: config.numClasses > 0 ? config.numClasses : DEFAULT_CONFIG.numClasses }, (_, i) => `Classe ${String.fromCharCode(65 + i)}`);

    if (userClassificationType) {
        context += `Tipo de Classificação Informado pelo Usuário: ${userClassificationType}\n\n`;
    }

    if (trainingMetrics && trainingMetrics.epochs.length > 0) {
      context += "### Métricas de Treinamento (Resumo da Última Época):\n";
      const lastEpochIndex = trainingMetrics.epochs.length - 1;
      context += `- Época Final: ${trainingMetrics.epochs[lastEpochIndex]}\n`;
      context += `- Perda Treino Final: ${trainingMetrics.trainLoss[lastEpochIndex]?.toFixed(4)}\n`;
      context += `- Perda Validação Final: ${trainingMetrics.validLoss[lastEpochIndex]?.toFixed(4)}\n`;
      context += `- Acurácia Treino Final: ${trainingMetrics.trainAcc[lastEpochIndex]?.toFixed(4)}\n`;
      context += `- Acurácia Validação Final: ${trainingMetrics.validAcc[lastEpochIndex]?.toFixed(4)}\n\n`;
    } else {
      context += "Nenhuma métrica de treinamento registrada.\n\n";
    }

    if (evaluationReport) {
      context += "### Relatório de Classificação:\n";
      context += `- Acurácia Geral: ${evaluationReport.accuracy?.toFixed(4)}\n`;
      effectiveClassNames.forEach(className => {
          const m = evaluationReport.classMetrics[className];
          if(m) context += `- Classe ${className}: Precisão=${m.precision.toFixed(3)}, Recall=${m.recall.toFixed(3)}, F1=${m.f1Score.toFixed(3)}, Especificidade=${m.specificity?.toFixed(3) ?? 'N/A'}, Suporte=${m.support}\n`;
      });
      context += `- Média Macro: Precisão=${evaluationReport.macroAvg.precision.toFixed(3)}, Recall=${evaluationReport.macroAvg.recall.toFixed(3)}, F1=${evaluationReport.macroAvg.f1Score.toFixed(3)}, Especificidade=${evaluationReport.macroAvg.specificity?.toFixed(3) ?? 'N/A'}\n`;
      context += `- Média Ponderada: Precisão=${evaluationReport.weightedAvg.precision.toFixed(3)}, Recall=${evaluationReport.weightedAvg.recall.toFixed(3)}, F1=${evaluationReport.weightedAvg.f1Score.toFixed(3)}, Especificidade=${evaluationReport.weightedAvg.specificity?.toFixed(3) ?? 'N/A'}\n`;
      if (evaluationReport.aucpr) {
        context += `- AUC-PR (Macro): ${evaluationReport.aucpr.toFixed(3)}\n`;
      }
      context += "\n";
    }

    if (confusionMatrix) {
      context += "### Matriz de Confusão (Normalizada):\n";
      context += `Classes: ${confusionMatrix.labels.join(', ')}\n`;
      confusionMatrix.matrix.forEach((row, i) => {
          context += `Real ${confusionMatrix.labels[i]}: Predito [${row.map(cell => cell.toFixed(2)).join(', ')}]\n`;
      });
      context += "\n";
    }
    
    if (errorAnalysisData && errorAnalysisData.length > 0) {
        context += "### Análise de Erros (Amostra de Imagens Mal Classificadas):\n";
        errorAnalysisData.slice(0, 3).forEach(item => { 
            context += `- Imagem (Placeholder): Real: ${item.trueLabel}, Predito: ${item.predLabel}\n`;
        });
        context += "\n";
    }

    if (clusterData) {
        context += "### Métricas de Clusterização:\n";
        context += `- Hierárquico: ARI=${clusterData.metrics.hierarchicalARI.toFixed(3)}, NMI=${clusterData.metrics.hierarchicalNMI.toFixed(3)}\n`;
        context += `- K-Means: ARI=${clusterData.metrics.kmeansARI.toFixed(3)}, NMI=${clusterData.metrics.kmeansNMI.toFixed(3)}\n\n`;
    }

    if (rocCurveData) {
      context += `### Curva ROC AUC: ${rocCurveData.auc.toFixed(3)}\n`;
    }
    if (prCurveData) {
      context += `### Curva PR AUC: ${prCurveData.auc.toFixed(3)}\n\n`;
    }

    if (!areResultsAvailable) {
      return "Nenhum resultado de modelo disponível para análise. Por favor, execute o treinamento primeiro.";
    }
    return context;
  }, [config, trainingMetrics, evaluationReport, confusionMatrix, classNamesInData, zipFile, rocCurveData, prCurveData, errorAnalysisData, clusterData, areResultsAvailable, userClassificationType]);


  const initializeChatIA = useCallback(async () => {
    if (!process.env.API_KEY) {
      console.warn("API Key for Gemini not found. Chat IA disabled.");
      setChatMessages([{ id: 'system-no-api', role: 'system', text: "Chave da API para Gemini não configurada (process.env.API_KEY não encontrada). O Chat IA está desabilitado.", timestamp: new Date() }]);
      return null;
    }
    if (geminiChatInstance && !awaitingClassificationType) return geminiChatInstance;

    setIsChatLoading(true);
    setSimulatedAgentLog([]);
    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      let resultsContext = getResultsAsTextContext();
      
      const systemInstruction = `Você é Marcelo Claro, um assistente especialista em IA e ciência de dados. Seu objetivo é analisar os resultados de um modelo de classificação de imagens e responder a perguntas sobre eles. Os resultados relevantes da execução atual e o tipo de classificação (se informado) são fornecidos abaixo. Seja claro, conciso e útil. Se os resultados não estiverem disponíveis ou forem insuficientes, informe o usuário que ele precisa treinar um modelo primeiro ou fornecer mais detalhes. Se o usuário perguntar algo que exija pesquisa externa, simule a ativação de "agentes de pesquisa" e, em seguida, forneça uma resposta abrangente com base no seu conhecimento e no contexto.
      Contexto dos Resultados:
      ${resultsContext}`;

      const newChat = ai.chats.create({
        model: 'gemini-2.5-flash-preview-04-17',
        config: { systemInstruction: systemInstruction },
      });
      setGeminiChatInstance(newChat);
      
      let initialMessage: UserOrModelMessage;
      if (!areResultsAvailable) {
        initialMessage = {
          id: crypto.randomUUID(), 
          role: 'model', 
          text: "Olá! Eu sou Marcelo Claro, seu assistente de IA. Estou aqui para ajudar a analisar os resultados do seu modelo (seja do processamento atual ou de dados em formato CSV que você tenha), discutir características das imagens, ou explicar conceitos de IA.\n\nNo momento, parece que nenhum resultado específico desta sessão foi carregado. Se você já realizou um processamento na aplicação, tente 'Recarregar Contexto'. Caso contrário, por favor, inicie o processamento para que eu possa analisar os dados gerados, ou podemos conversar sobre IA em geral!", 
          timestamp: new Date()
        };
      } else if (!userClassificationType) {
        initialMessage = {
          id: crypto.randomUUID(), role: 'model', text: "Olá! Eu sou Marcelo Claro, seu assistente de IA. Para te ajudar melhor, qual é o tipo de classificação de imagens que você está realizando com este dataset (ex: diagnóstico de melanoma, identificação de tipos de rochas, controle de qualidade industrial, etc.)?", timestamp: new Date()
        };
        setAwaitingClassificationType(true);
      } else {
         initialMessage = {
           id: crypto.randomUUID(), role: 'model', text: `Olá! Com base nos resultados e no seu foco em "${userClassificationType}", como posso te ajudar a analisar o desempenho do modelo hoje?`, timestamp: new Date()
         };
      }
      
      setChatMessages(prev => {
          // Avoid adding duplicate initial messages
          if (prev.length === 0 || prev[prev.length -1].id.startsWith('system-')) {
              return [initialMessage];
          }
          // If results became available and chat was already open, re-prompt if needed
          if (areResultsAvailable && !userClassificationType && !prev.some(m => m.text.includes("qual é o tipo de classificação"))) {
              setAwaitingClassificationType(true);
              return [...prev, initialMessage];
          }
          return prev;
      });

      return newChat;
    } catch (error) {
      console.error("Failed to initialize Gemini chat:", error);
      setChatMessages([{ id: 'system-error-init', role: 'system', text: "Falha ao inicializar o chat com IA. Verifique o console para erros.", timestamp: new Date() }]);
      return null;
    } finally {
      setIsChatLoading(false);
    }
  }, [getResultsAsTextContext, geminiChatInstance, areResultsAvailable, userClassificationType, awaitingClassificationType]);

  useEffect(() => {
    if (currentSection === AppSection.CHAT_IA && process.env.API_KEY && (!geminiChatInstance || (areResultsAvailable && !userClassificationType && !awaitingClassificationType))) {
      initializeChatIA();
    }
  }, [currentSection, geminiChatInstance, initializeChatIA, areResultsAvailable, userClassificationType, awaitingClassificationType]);


  const handleSendMessageToChatIA = async () => {
    if (!chatInput.trim()) return;
    
    const userMessage: UserOrModelMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      text: chatInput,
      timestamp: new Date(),
    };
    setChatMessages(prev => [...prev, userMessage]);
    const currentChatInput = chatInput;
    setChatInput('');
    setIsChatLoading(true);
    setSimulatedAgentLog([]); // Clear previous agent log

    let currentChat = geminiChatInstance;
    if (!currentChat) {
        currentChat = await initializeChatIA(); 
    }
    
    if (!currentChat) {
       const errorMessage: UserOrModelMessage = {
          id: crypto.randomUUID(),
          role: 'model',
          text: "Desculpe, não consigo processar sua mensagem. O chat não foi inicializado corretamente. Verifique se a API Key está configurada e se os resultados do modelo estão disponíveis (após treinamento).",
          timestamp: new Date()
      };
      setChatMessages(prev => [...prev, errorMessage]);
      setIsChatLoading(false);
      return;
    }

    // Handle setting classification type
    if (awaitingClassificationType) {
      setUserClassificationType(currentChatInput);
      setAwaitingClassificationType(false);
      const confirmationMsg: UserOrModelMessage = {
        id: crypto.randomUUID(),
        role: 'model',
        text: `Entendido! Foco em "${currentChatInput}". Agora, como posso te ajudar com a análise dos resultados?`,
        timestamp: new Date()
      };
      setChatMessages(prev => [...prev, confirmationMsg]);
      // Re-initialize chat with new context for system prompt if needed or update existing
       const updatedContextInstruction = `Você é Marcelo Claro... Contexto dos Resultados (incluindo tipo de classificação: ${currentChatInput}): ${getResultsAsTextContext()}`;
       // This is a simplified way; a more robust Gemini SDK usage might involve sending new history.
       // For now, the next Gemini call will use the updated context via getResultsAsTextContext() in the system prompt of a *new* chat instance if we re-create, or just the updated context in memory.
       // Let's re-initialize for simplicity to update system prompt effectively for next turn.
       setGeminiChatInstance(null); // Force re-init on next send or auto-init
       await initializeChatIA(); // Re-initialize with the new classification type in context
       setIsChatLoading(false);
       return;
    }

    // Simulate agent research for complex queries
    const researchKeywords = ["artigos", "pesquisa", "avanços", "multidisciplinar", "literatura", "estudos recentes", "tendências em"];
    const needsResearch = researchKeywords.some(kw => currentChatInput.toLowerCase().includes(kw));

    if (needsResearch && userClassificationType) {
        const agentLogUpdates: string[] = [];
        agentLogUpdates.push(`[${new Date().toLocaleTimeString()}] INFO: Consulta do usuário sugere necessidade de pesquisa aprofundada.`);
        agentLogUpdates.push(`[${new Date().toLocaleTimeString()}] AGENT_SYSTEM: Ativando Agente de Pesquisa Especializado em "${userClassificationType}".`);
        setSimulatedAgentLog(prev => [...prev, ...agentLogUpdates]);
        
        await new Promise(resolve => setTimeout(resolve, 800)); // Simulate delay
        agentLogUpdates.length = 0; // Clear for next batch
        agentLogUpdates.push(`[${new Date().toLocaleTimeString()}] AGENT_WEB_QUERY: Buscando artigos e dados sobre "IA para ${userClassificationType}" e "${currentChatInput.substring(0,30)}...".`);
        setSimulatedAgentLog(prev => [...prev, ...agentLogUpdates]);

        await new Promise(resolve => setTimeout(resolve, 1200));
        agentLogUpdates.length = 0;
        agentLogUpdates.push(`[${new Date().toLocaleTimeString()}] AGENT_ANALYSIS: Processando e sintetizando informações de múltiplas fontes...`);
        setSimulatedAgentLog(prev => [...prev, ...agentLogUpdates]);
        
        await new Promise(resolve => setTimeout(resolve, 1000));
        agentLogUpdates.length = 0;
        agentLogUpdates.push(`[${new Date().toLocaleTimeString()}] AGENT_SYSTEM: Síntese concluída. Preparando resposta...`);
        setSimulatedAgentLog(prev => [...prev, ...agentLogUpdates]);
    }


    try {
      // Ensure chat is re-initialized if it was reset (e.g. after setting classification type)
      if (!geminiChatInstance) {
          currentChat = await initializeChatIA();
          if (!currentChat) throw new Error("Falha ao re-inicializar o chat.");
      }
      const response = await currentChat.sendMessage({ message: currentChatInput });
      const modelResponse: UserOrModelMessage = {
        id: crypto.randomUUID(),
        role: 'model',
        text: response.text,
        timestamp: new Date(),
      };
      setChatMessages(prev => [...prev, modelResponse]);
    } catch (error) {
      console.error("Error sending message to Gemini:", error);
      const errorMessage: UserOrModelMessage = {
        id: crypto.randomUUID(),
        role: 'model',
        text: "Desculpe, ocorreu um erro ao tentar obter uma resposta da IA. Verifique o console para detalhes.",
        timestamp: new Date(),
      };
      setChatMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsChatLoading(false);
    }
  };


  const startTraining = useCallback(() => {
    if (!zipFile) {
      alert("Por favor, carregue um arquivo ZIP primeiro.");
      return;
    }

    const currentEffectiveNumClasses = config.fineTune ? numClassesInData : config.numClasses;
    const currentClassNames = config.fineTune && classNamesInData.length > 0 ? classNamesInData : Array.from({ length: config.numClasses > 0 ? config.numClasses: DEFAULT_CONFIG.numClasses }, (_, i) => `Classe ${String.fromCharCode(65 + i)}`);

    if (currentEffectiveNumClasses <=0) {
        alert("Número de classes inválido. Verifique os dados carregados ou a configuração.");
        return;
    }
    
    const effectiveConfig = { ...config, numClasses: currentEffectiveNumClasses };

    if (effectiveConfig.fineTune && effectiveConfig.numClasses !== numClassesInData) {
      alert(`Aviso: 'Fine-Tuning Completo' está ativo. O número de classes (${numClassesInData}) e os nomes das classes foram determinados pelo arquivo ZIP. A configuração de 'Número de Classes' no painel foi ajustada.`);
    }
    
    setIsTraining(true);
    setIsTrainingComplete(false);
    setAreResultsAvailable(false);
    setGeminiChatInstance(null); 
    setChatMessages([]);
    setUserClassificationType(null); 
    setAwaitingClassificationType(false);
    setSimulatedAgentLog([]);
    
    const newLog: string[] = ["INFO: Iniciando processo de treinamento do modelo..."];
    newLog.push("INFO: Parâmetros de Configuração Aplicados:");
    (Object.keys(effectiveConfig) as Array<keyof SidebarConfig>).forEach(key => {
        let prettyKey = key.replace(/([A-Z])/g, ' $1'); 
        prettyKey = prettyKey.charAt(0).toUpperCase() + prettyKey.slice(1); 
        
        const configKey = key as keyof SidebarConfig;
        const value: SidebarConfig[typeof configKey] = effectiveConfig[configKey];
        let displayValue: string;

        if (typeof value === 'boolean') {
            displayValue = value ? 'Sim' : 'Não';
        } else if (value === null || typeof value === 'undefined') {
            displayValue = 'Não definido';
        } else {
            displayValue = String(value);
        }
        
        if (SIDEBAR_EXPLANATIONS.hasOwnProperty(key)) {
             const explanationTitle = SIDEBAR_EXPLANATIONS[key as keyof typeof SIDEBAR_EXPLANATIONS].split('\n')[0].replace('O que é: ', '').replace('Como afeta: ', '').split('.')[0];
             if (explanationTitle.length < 40 && !explanationTitle.toLowerCase().includes('o que é')) { 
                 prettyKey = explanationTitle;
             }
        }
        newLog.push(`INFO:  - ${prettyKey}: ${displayValue}`);
    });
    newLog.push("INFO: ---");
    setTrainingLog(newLog);

    setTrainingStatus({ currentEpoch: 0, totalEpochs: effectiveConfig.epochs, message: "INFO: Inicializando ambiente de treinamento..." });
    
    const initialMetrics: TrainingMetrics = { epochs: [], trainLoss: [], validLoss: [], trainAcc: [], validAcc: [] };
    setTrainingMetrics(initialMetrics);
    setEvaluationReport(null); setConfusionMatrix(null); setErrorAnalysisData(null); setClusterData(null);
    setAugmentedEmbeddingsData(null); setRocCurveData(null); setPrCurveData(null);
    setCurrentSection(AppSection.TRAINING);

    let epoch = 0;
    const epochMetrics = { ...initialMetrics }; 
    let bestValidLoss = Infinity;
    let epochsNoImprove = 0;

    const interval = setInterval(() => {
      epoch++;
      const currentEpochLog = [...trainingLog]; 

      const newTrainLoss = 1 / Math.log10(epoch + 1) + Math.random() * 0.2;
      const newValidLoss = 1 / Math.log10(epoch + 1) + 0.1 + Math.random() * 0.2;
      const newTrainAcc = Math.min(0.95, 0.5 + Math.log(epoch) * 0.1 + Math.random() * 0.1);
      const newValidAcc = Math.min(0.90, 0.45 + Math.log(epoch) * 0.1 + Math.random() * 0.1);

      epochMetrics.epochs.push(epoch);
      epochMetrics.trainLoss.push(newTrainLoss);
      epochMetrics.validLoss.push(newValidLoss);
      epochMetrics.trainAcc.push(newTrainAcc);
      epochMetrics.validAcc.push(newValidAcc);
      
      setTrainingMetrics({ ...epochMetrics });
      setTrainingStatus({ currentEpoch: epoch, totalEpochs: effectiveConfig.epochs, message: `INFO: Época ${epoch}/${effectiveConfig.epochs} em processamento...` });
      currentEpochLog.push(`DEBUG: Época ${epoch}: Perda Treino: ${newTrainLoss.toFixed(4)}, Acc Treino: ${newTrainAcc.toFixed(4)}, Perda Val: ${newValidLoss.toFixed(4)}, Acc Val: ${newValidAcc.toFixed(4)}`);

      if (newValidLoss < bestValidLoss) {
        bestValidLoss = newValidLoss;
        epochsNoImprove = 0;
        currentEpochLog.push(`INFO: Época ${epoch}: Nova melhor perda de validação: ${bestValidLoss.toFixed(4)}.`);
      } else {
        epochsNoImprove++;
        currentEpochLog.push(`INFO: Época ${epoch}: Perda de validação (${newValidLoss.toFixed(4)}) não melhorou. Melhor: ${bestValidLoss.toFixed(4)}. Sem melhora por ${epochsNoImprove} épocas.`);
      }
      setTrainingLog(currentEpochLog); 

      const finishTraining = () => {
        clearInterval(interval);
        setIsTraining(false);
        setIsTrainingComplete(true);
        
        const generatedReport = generateMockClassificationReport(currentClassNames);
        const generatedCM = generateMockConfusionMatrix(currentClassNames);
        const generatedErrorAnalysis = generateMockErrorAnalysis(currentClassNames, sampleImagesFromZip);
        const generatedClusterData = generateMockClusterData(currentEffectiveNumClasses > 0 ? currentEffectiveNumClasses : 2, currentClassNames);
        const generatedAugmentedEmbeddings = generateMockAugmentedEmbeddings(50, 3, currentClassNames);
        const generatedRocData = generateMockROCCurveData(currentClassNames);
        const generatedPrData = generateMockPRCurveData(currentClassNames);

        setEvaluationReport(generatedReport); setConfusionMatrix(generatedCM); setErrorAnalysisData(generatedErrorAnalysis);
        setClusterData(generatedClusterData); setAugmentedEmbeddingsData(generatedAugmentedEmbeddings);
        setRocCurveData(generatedRocData); setPrCurveData(generatedPrData);
        
        setAreResultsAvailable(true);
        setCurrentSection(AppSection.EVALUATION);
        if (process.env.API_KEY) {
             initializeChatIA();
        }
      };

      if (epochsNoImprove >= effectiveConfig.patience) {
        const earlyStopMessage = `INFO: Processamento interrompido por Parada Antecipada na época ${epoch}. Paciência (${effectiveConfig.patience}) atingida.`;
        setTrainingStatus({ currentEpoch: epoch, totalEpochs: effectiveConfig.epochs, message: earlyStopMessage });
        setTrainingLog(prevLog => [...prevLog, "INFO: ---", earlyStopMessage, "INFO: Gerando resultados finais..."]);
        finishTraining();
        return; 
      }

      if (epoch >= effectiveConfig.epochs) { 
        const completionMessage = "INFO: Processamento completo (todas as épocas concluídas)!";
        setTrainingStatus({ currentEpoch: epoch, totalEpochs: effectiveConfig.epochs, message: completionMessage });
        setTrainingLog(prevLog => [...prevLog, "INFO: ---", completionMessage, "INFO: Gerando resultados finais..."]);
        finishTraining();
      }
    }, 700); 
  }, [zipFile, config, numClassesInData, classNamesInData, trainingLog, sampleImagesFromZip, initializeChatIA, getResultsAsTextContext]); 

  const handleImageForEvalUpload = (file: File) => {
    setUploadedImageForEval(file);
    const reader = new FileReader();
    const currentClassNames = config.fineTune && zipFile && classNamesInData.length > 0 ? classNamesInData : Array.from({ length: config.numClasses > 0 ? config.numClasses : DEFAULT_CONFIG.numClasses }, (_, i) => `Classe ${String.fromCharCode(65 + i)}`);
    
    reader.onloadend = async () => {
        const predClass = currentClassNames.length > 0 ? currentClassNames[Math.floor(Math.random() * currentClassNames.length)] : "Classe Indefinida";
        let uncertainty: number | undefined = undefined;
        if (config.simulatedUncertainty) {
            uncertainty = generateMockUncertaintyScore();
        }

      setIndividualEval({
        imageSrc: reader.result as string,
        predictedClass: predClass,
        confidence: Math.random() * 0.5 + 0.5,
        uncertaintyScore: uncertainty, 
      });
      try {
        const camDataUrl = await generateMockCAMImage(reader.result as string, config.camMethod);
        setCamImage(camDataUrl);
      } catch (error) {
        console.error("Erro ao gerar imagem CAM:", error);
        setCamImage(null); 
      }
    };
    reader.readAsDataURL(file);
    setCurrentSection(AppSection.INSPECTOR);
  };

  useEffect(() => {
    // Effect for general config changes, if any specific action needed.
  }, [config.numClasses, config.fineTune, zipFile]);


  const renderSection = () => {
    const effectiveNumClasses = config.fineTune && zipFile ? numClassesInData : config.numClasses;
    const effectiveClassNames = config.fineTune && zipFile && classNamesInData.length > 0 ? classNamesInData : Array.from({ length: effectiveNumClasses > 0 ? effectiveNumClasses : DEFAULT_CONFIG.numClasses }, (_, i) => `Classe ${String.fromCharCode(65 + i)}`);

    switch (currentSection) {
      case AppSection.DATA_CONFIG:
        return <DataUpload 
                  onFileUpload={handleZipUpload} 
                  numClasses={numClassesInData > 0 ? numClassesInData : config.numClasses} 
                  classNames={classNamesInData.length > 0 ? classNamesInData : Array.from({ length: config.numClasses > 0 ? config.numClasses : DEFAULT_CONFIG.numClasses }, (_, i) => `Classe Padrão ${String.fromCharCode(65 + i)}`)}
                  sampleImages={sampleImagesFromZip}
                  isProcessingZip={isProcessingZip}
                  zipFileLoaded={zipFile !== null}
               />;
      case AppSection.TRAINING:
        return trainingStatus ? (
            <TrainingMonitor 
                status={trainingStatus} 
                metrics={trainingMetrics} 
                isTrainingComplete={isTrainingComplete}
                onExportMetrics={handleExportTrainingMetrics}
                trainingLog={trainingLog} 
            />
        ) : <p className="text-center text-gray-400 py-10">Processamento não iniciado. Configure e inicie na barra lateral.</p>;
      case AppSection.EVALUATION:
        return <EvaluationResults 
                  report={evaluationReport} 
                  confusionMatrix={confusionMatrix} 
                  errorAnalysis={errorAnalysisData} 
                  classNames={effectiveClassNames}
                  rocCurveData={rocCurveData}
                  prCurveData={prCurveData} 
                  isTrainingComplete={isTrainingComplete} 
                />;
      case AppSection.CLUSTERING:
        return clusterData || augmentedEmbeddingsData ? (
            <ClusteringAnalysis 
                data={clusterData} 
                augmentedEmbeddings={augmentedEmbeddingsData}
                classNames={effectiveClassNames} 
            />
        ) : <p className="text-center text-gray-400 py-10">Análise de clusterização não disponível. Conclua o processamento primeiro.</p>;
      case AppSection.INSPECTOR:
        return <ImageInspector 
                  evaluation={individualEval} 
                  camImage={camImage} 
                  onImageUpload={handleImageForEvalUpload} 
                />;
      case AppSection.CHAT_IA:
        return <ChatIA
                  messages={chatMessages}
                  inputValue={chatInput}
                  onInputChange={(e) => setChatInput(e.target.value)}
                  onSendMessage={handleSendMessageToChatIA}
                  isLoading={isChatLoading}
                  onRetryInit={initializeChatIA}
                  isReady={!!process.env.API_KEY}
                  resultsAvailable={areResultsAvailable}
                  simulatedAgentLog={simulatedAgentLog}
               />;
      case AppSection.EXPLANATIONS:
        return <ExplanationHub topics={ALL_EXPLANATIONS} />;
      default:
        return <p className="text-center text-gray-400 py-10">Selecione uma seção no menu de navegação.</p>;
    }
  };

  return (
    <div className="flex h-screen font-sans">
      <Sidebar 
        config={config} 
        onConfigChange={handleConfigChange} 
        onStartTraining={startTraining} 
        onSaveConfig={saveConfigToJson} 
        isTraining={isTraining}
        zipFileLoaded={zipFile !== null} 
      />
      
      <main className="flex-1 p-6 overflow-y-auto bg-gray-850 flex flex-col">
        <header className="mb-6">
          <div className="flex justify-between items-center">
            <div className="flex items-center">
                <div>
                    <h1 className="text-3xl lg:text-4xl font-bold text-blue-400">Geomaker AI - Analisador de Imagens DL</h1>
                    <p className="text-gray-400 text-sm lg:text-base">Plataforma Interativa para Análise de Classificação e Clusterização de Imagens.</p>
                </div>
            </div>
            {areResultsAvailable && (
                 <button
                    onClick={handleExportResults}
                    title="Baixar todos os resultados e métricas em um arquivo CSV."
                    className="px-4 py-2 border border-blue-500 text-blue-400 rounded-md shadow-sm text-sm font-medium hover:bg-blue-500 hover:text-gray-900 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-blue-600 transition-colors"
                >
                    <i className="fas fa-download mr-2"></i> Baixar Resultados (CSV)
                </button>
            )}
          </div>
        </header>

        <nav className="mb-6">
          <ul className="flex space-x-1 sm:space-x-2 border-b-2 border-gray-700 overflow-x-auto pb-px">
            {Object.values(AppSection).map((section) => (
              <li key={section} className="flex-shrink-0">
                <button
                  onClick={() => setCurrentSection(section)}
                  className={`px-3 py-2 sm:px-4 font-medium rounded-t-lg transition-colors duration-150 text-xs sm:text-sm
                    ${currentSection === section ? 'bg-blue-600 text-gray-900' : 'text-gray-300 hover:bg-gray-700 hover:text-blue-400'}`}
                >
                  {section}
                </button>
              </li>
            ))}
          </ul>
        </nav>

        <div className="flex-grow bg-gray-850 p-4 sm:p-6 rounded-lg shadow-xl min-h-[50vh]">
          {renderSection()}
        </div>
       
        <footer className="mt-auto pt-8 text-center text-sm text-gray-500 pb-6"> 
            <p className="text-xs">Atenção: O pipeline principal de treinamento de modelos e geração de resultados é representativo e opera no frontend. O Chat com IA (Marcelo Claro) utiliza a API Gemini para respostas reais. Para pesquisa e diagnóstico com o pipeline principal, seria necessária uma implementação com backend dedicado. Veja "Aprender Conceitos &gt; Arquitetura da Aplicação" para mais detalhes.</p>
        </footer>
      </main>
    </div>
  );
};

export default App;
