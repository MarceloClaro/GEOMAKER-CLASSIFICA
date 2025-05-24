
import React, { useState } from 'react';
import type { ExplanationTopic } from '../types';
import { GoogleGenAI } from "@google/genai";

interface ExplanationHubProps {
  topics: ExplanationTopic[];
}

const API_KEY = process.env.API_KEY;
let ai: GoogleGenAI | null = null;
if (API_KEY) {
  ai = new GoogleGenAI({ apiKey: API_KEY });
}

const ExplanationCard: React.FC<{ topic: ExplanationTopic }> = ({ topic }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [aiExplanation, setAiExplanation] = useState<string | null>(null);
  const [isLoadingAi, setIsLoadingAi] = useState(false);

  const fetchAiExplanation = async () => {
    if (!ai) {
      setAiExplanation("API Gemini não configurada. Não é possível buscar explicação da IA.");
      return;
    }
    setIsLoadingAi(true);
    setAiExplanation(null);
    try {
      const model = 'gemini-2.5-flash-preview-04-17';
      const prompt = topic.promptSuggestion || `Explique "${topic.title}" no contexto de aprendizado profundo para análise de imagens. Foque nestas palavras-chave: ${topic.keywords.join(', ')}. Mantenha conciso e fácil de entender.`;
      
      const result = await ai.models.generateContent({
        model: model,
        contents: prompt,
      });
      // FIX: Access text directly from result object as per Gemini API guidelines
      const responseText = result.text;
      setAiExplanation(responseText);

    } catch (error) {
      console.error("Erro ao buscar explicação da IA:", error);
      setAiExplanation("Desculpe, não foi possível buscar uma explicação da IA no momento.");
    } finally {
      setIsLoadingAi(false);
    }
  };

  return (
    <div className="bg-gray-800 p-5 rounded-lg shadow-xl mb-6"> {/* Card for each topic */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full text-left flex justify-between items-center py-2 text-xl font-semibold text-blue-300 hover:text-blue-200 transition-colors"
        aria-expanded={isOpen}
        aria-controls={`explanation-${topic.id}`}
      >
        {topic.title}
        <i className={`fas fa-chevron-down transform transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}></i>
      </button>
      {isOpen && (
        <div id={`explanation-${topic.id}`} className="mt-3 space-y-3 text-gray-300 text-sm">
          <p className="leading-relaxed">{topic.summary}</p>
          {topic.details && <p className="border-t border-gray-700 pt-3 mt-3 leading-relaxed">{topic.details}</p>}
          
          <button 
            onClick={fetchAiExplanation}
            disabled={isLoadingAi || !ai}
            className="mt-3 px-4 py-2 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded-md shadow-md disabled:opacity-60 flex items-center transition-colors"
          >
            {isLoadingAi ? (
              <><i className="fas fa-spinner fa-spin mr-2"></i>Carregando da IA...</>
            ) : (
              <><i className="fas fa-robot mr-2"></i>Perguntar à IA</>
            )}
          </button>
          
          {aiExplanation && (
            <div className="mt-4 p-4 bg-gray-750 border border-gray-600 rounded-md shadow-inner">
              <h4 className="font-semibold text-blue-400 mb-2 text-base">Explicação Detalhada da IA:</h4>
              <p className="whitespace-pre-wrap leading-relaxed">{aiExplanation}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export const ExplanationHub: React.FC<ExplanationHubProps> = ({ topics }) => {
  return (
    <div className="space-y-8"> 
      <h2 className="text-3xl font-bold text-blue-400 mb-4 text-center">Aprenda Conceitos Chave</h2>
      <p className="text-gray-400 mb-6 text-center max-w-2xl mx-auto">Expanda os tópicos abaixo para uma breve explicação. Você também pode solicitar mais detalhes à nossa IA para um aprofundamento sobre cada conceito.</p>
      {!ai && <p className="text-yellow-400 text-sm bg-yellow-900 bg-opacity-50 p-3 rounded-md text-center mb-6 border border-yellow-700"><i className="fas fa-exclamation-triangle mr-2"></i>Chave da API Gemini não configurada. Explicações via IA estão desabilitadas.</p>}
      <div className="space-y-4">
        {topics.map(topic => (
          <ExplanationCard key={topic.id} topic={topic} />
        ))}
      </div>
    </div>
  );
};