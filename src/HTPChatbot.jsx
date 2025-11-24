import React, { useState, useRef, useEffect } from "react";
// 필요한 아이콘을 htp_chatbot_251116.tsx와 유사하게 조정
import { Upload, Eraser, MessageCircle, Brain, Pencil } from "lucide-react"; 

// RunPod URL을 여기에 입력하세요 (예: https://your-pod-id.proxy.runpod.net)
const API_BASE = process.env.REACT_APP_API_BASE || "http://127.0.0.1:8000";

// 탭 이름 매핑
const tabNames = {
    house: { ko: '집', en: 'House' },
    tree: { ko: '나무', en: 'Tree' },
    person: { ko: '사람', en: 'Person' }
};

const HTPChatbot = () => {
  const [currentPage, setCurrentPage] = useState(1);
  const [activeTab, setActiveTab] = useState("house");

  const [drawings, setDrawings] = useState({
    house: null,
    tree: null,
    person: null,
  });

  const [context, setContext] = useState(null);
  const canvasRef = useRef(null);

  const [isDrawing, setIsDrawing] = useState(false);
  const [brushSize, setBrushSize] = useState(2); 
  const [brushColor, setBrushColor] = useState("#000000");

  const [captions, setCaptions] = useState({
    house: "",
    tree: "",
    person: "",
  });

  const [interpretations, setInterpretations] = useState({
    house: "",
    tree: "",
    person: "",
  });

  const [finalInterpretation, setFinalInterpretation] = useState("");
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const [questionCount, setQuestionCount] = useState(0);
  const [isComplete, setIsComplete] = useState(false);
  const chatEndRef = useRef(null);
  const [isLoading, setIsLoading] = useState(false); 

  /** --------------------------
   * Canvas 초기화
   * -------------------------- */
  useEffect(() => {
    if (canvasRef.current) {
      const c = canvasRef.current;
      const ctx = c.getContext("2d");
      ctx.lineWidth = brushSize;
      ctx.strokeStyle = brushColor;
      ctx.lineCap = "round";
      setContext(ctx);

      ctx.clearRect(0, 0, c.width, c.height);
      if (drawings[activeTab]) {
        const img = new Image();
        img.onload = () => ctx.drawImage(img, 0, 0);
        img.src = drawings[activeTab];
      } else {
        // 새 그림판에 기본 안내 텍스트 추가
        ctx.font = '24px Arial';
        ctx.fillStyle = '#9ca3af';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${tabNames[activeTab].ko}을(를) 그려주세요.`, c.width / 2, c.height / 2);
      }
    }
  }, [activeTab, brushSize, brushColor, drawings]);

  /** --------------------------
   * Canvas 그리기 및 저장
   * -------------------------- */
  const getCoords = (e) => {
    const c = canvasRef.current;
    const rect = c.getBoundingClientRect();
    const clientX = e.clientX || (e.touches && e.touches[0].clientX);
    const clientY = e.clientY || (e.touches && e.touches[0].clientY);
    
    // 단순화된 좌표 계산 유지
    return {
      x: clientX - rect.left,
      y: clientY - rect.top,
    };
  };

  const startDraw = (e) => {
    if (!context) return;
    setIsDrawing(true);
    const { x, y } = getCoords(e);
    context.beginPath();
    context.moveTo(x, y);
    context.lineTo(x + 0.5, y + 0.5); 
    context.stroke();
  };

  const draw = (e) => {
    if (!isDrawing || !context) return;
    const { x, y } = getCoords(e);
    context.lineTo(x, y);
    context.stroke();
  };

  const endDraw = () => {
    if (!context) return;
    setIsDrawing(false);
    if (canvasRef.current) {
      const data = canvasRef.current.toDataURL();
      setDrawings((prev) => ({ ...prev, [activeTab]: data }));
    }
  };
  
  const saveCurrentDrawing = () => {
    if (canvasRef.current) {
      const data = canvasRef.current.toDataURL();
      setDrawings((prev) => ({ ...prev, [activeTab]: data }));
    }
  };

  const clearCanvas = () => {
    if (!context || !canvasRef.current) return;
    context.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    setDrawings((prev) => ({ ...prev, [activeTab]: null }));
    
    const c = canvasRef.current;
    context.font = '24px Arial';
    context.fillStyle = '#9ca3af';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillText(`${tabNames[activeTab].ko}을(를) 그려주세요.`, c.width / 2, c.height / 2);
  };

  /** --------------------------
   * 이미지 업로드
   * -------------------------- */
  const uploadImage = (e) => {
    const file = e.target.files[0];
    if (!file || !context || !canvasRef.current) return;

    const reader = new FileReader();
    reader.onload = (ev) => {
      const img = new Image();
      img.onload = () => {
        const c = canvasRef.current;
        context.clearRect(0, 0, c.width, c.height);
        // 이미지를 캔버스 크기에 맞게 조정하여 그리기
        context.drawImage(img, 0, 0, c.width, c.height); 
        setDrawings((prev) => ({
          ...prev,
          [activeTab]: c.toDataURL(),
        }));
      };
      img.src = ev.target.result;
    };
    reader.readAsDataURL(file);
    e.target.value = null; 
  };
  
  const hasAnyDrawing = () => {
    return drawings.house || drawings.tree || drawings.person;
  };

  /** --------------------------
   * 페이지 1 → 2 : 캡션 + RAG + 해석
   * -------------------------- */
  const runInterpretation = async () => {
    const tabs = ["house", "tree", "person"];
    const drawnTabs = tabs.filter(t => drawings[t]);

    if (drawnTabs.length === 0) {
      alert("적어도 하나의 그림을 그려주세요.");
      return;
    }
    
    setIsLoading(true);
    // 초기 로딩 메시지는 htp_chatbot_251116.tsx의 로직을 따르지 않고 기존 로직을 따름
    setMessages([{ role: "assistant", content: "그림 분석 중입니다. 잠시만 기다려주세요..." }]);

    let newCaptions = {};
    let newInterps = {};

    for (const t of drawnTabs) {
      const base64 = drawings[t].split(",")[1];

      try {
        // 1) 캡션 생성
        const capRes = await fetch(`${API_BASE}/caption`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_base64: base64 }),
        });
        
        const capJson = await capRes.json();
        
        // capJson.caption: 문자열(JSON) → 객체로 파싱
        const captionObj = JSON.parse(capJson.caption); // { ko, en }
        
        // ✅ 상태에는 한국어 문자열만 저장
        newCaptions[t] = captionObj.ko;
        
        // ✅ RAG에도 한국어만 전달
        const ragRes = await fetch(`${API_BASE}/rag`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            caption: captionObj.ko,
            image_type: t,
          }),
        });
        const ragJson = await ragRes.json();
        console.log(captionObj.en)
        // 3) 개별 해석
        const intRes = await fetch(`${API_BASE}/interpret_single`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            caption: captionObj.en,
            rag_docs: ragJson.documents,
            image_type: t,
          }),
        });
        const intJson = await intRes.json();
        newInterps[t] = intJson.interpretation;
        
      } catch (error) {
        console.error(`Interpretation failed for ${t}:`, error);
        newCaptions[t] = `ERROR: 캡션 생성 실패`;
        newInterps[t] = `그림 해석 중 오류 발생. API 서버 상태 확인이 필요함.`;
      }
    }

    setCaptions(newCaptions);
    setInterpretations(newInterps);
    setCurrentPage(2);
    setIsLoading(false); // 분석 완료
    
    // 초기 챗봇 메시지 설정 (API 호출은 생략하고 텍스트만)
    setMessages([
      { role: "assistant", content: `그림 분석이 완료됨. 이제 추가로 궁금한 점을 최대 ${5 - questionCount}번까지 질문 가능함.` },
    ]);
  };

  /** --------------------------
   * 추가 질문
   * -------------------------- */
  const sendMessage = async () => {
    if (!inputMessage.trim() || isComplete || isLoading) return;

    const userMsg = inputMessage;
    setMessages((m) => [...m, { role: "user", content: userMsg }]);
    setInputMessage("");

    setIsLoading(true);
    
    // 로딩 메시지 설정
    const loadingText = questionCount < 5 && !isComplete ? "생각 중..." : "최종 해석 생성 중...";
    setMessages((m) => [...m, { role: "assistant", content: loadingText }]);
    const loadingMessageIndex = messages.length + 1; 

    try {
      if (questionCount < 5) {
        // 질문 응답
        const res = await fetch(`${API_BASE}/questions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            conversation: messages.concat({ role: "user", content: userMsg }),
          }),
        });
        const json = await res.json();
        
        // 로딩 메시지를 실제 답변으로 대체
        setMessages((m) => {
          const newMessages = m.slice(0, loadingMessageIndex);
          newMessages.push({ role: "assistant", content: json.question });
          return newMessages;
        });
        setQuestionCount((q) => q + 1);
        
      } else {
        // 최종 해석 요청
        const res = await fetch(`${API_BASE}/interpret_final`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            single_results: interpretations,
            conversation: messages.concat({ role: "user", content: userMsg }),
          }),
        });
        const json = await res.json();
        setFinalInterpretation(json.final);
        setIsComplete(true);
        
        // 로딩 메시지를 최종 해석 안내로 대체
        setMessages((m) => {
          const newMessages = m.slice(0, loadingMessageIndex);
          newMessages.push({ role: "assistant", content: "최종 해석이 완성되었음. 위의 최종 결과 섹션을 확인하세요." });
          return newMessages;
        });
      }
    } catch (error) {
       console.error("Chatbot API Error:", error);
       // 오류 발생 시 로딩 메시지를 오류 메시지로 대체
       setMessages((m) => {
          const newMessages = m.slice(0, loadingMessageIndex);
          newMessages.push({ role: "assistant", content: "오류 발생. 챗봇 API 서버 상태 확인이 필요함." });
          return newMessages;
        });
    } finally {
        setIsLoading(false);
    }
  };


  useEffect(() => {
    // 메시지 스크롤
    if (chatEndRef.current)
      chatEndRef.current.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // 페이지 1 디자인
  if (currentPage === 1) {
    return (
      <div className="h-screen bg-gradient-to-br from-blue-50 to-purple-50 p-6 overflow-hidden">
        <div className="max-w-5xl mx-auto h-full flex flex-col">
          <div className="text-center mb-4">
            <h1 className="text-4xl font-bold text-purple-600 mb-1">챗쪽이</h1>
            <h2 className="text-xl text-gray-700">HTP 심리 검사 해석 챗봇</h2>
          </div>

          <div className="bg-white rounded-2xl shadow-xl p-5 flex-1 flex flex-col overflow-hidden">
            {/* 탭 버튼 */}
            <div className="flex gap-2 mb-4">
              {['house', 'tree', 'person'].map((tab) => (
                <button
                  key={tab}
                  onClick={() => {
                    saveCurrentDrawing();
                    setActiveTab(tab);
                  }}
                  className={`flex-1 py-2 rounded-lg font-semibold transition-all ${
                    activeTab === tab
                      ? 'bg-purple-600 text-white shadow-md'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  <div className="flex flex-col items-center">
                    <span className="text-sm">{tabNames[tab].ko}</span>
                    <span className="text-xs opacity-75">{tabNames[tab].en}</span>
                  </div>
                  {drawings[tab] && (
                    <span className="ml-2 text-xs">✓</span>
                  )}
                </button>
              ))}
            </div>

            {/* 브러시 컨트롤 */}
            <div className="mb-3 flex gap-4 items-center bg-gray-50 p-3 rounded-lg">
              <div className="flex items-center gap-2">
                <label className="text-xs font-medium text-gray-700">브러쉬 크기:</label>
                <input
                  type="range"
                  min="1"
                  max="20"
                  value={brushSize}
                  onChange={(e) => setBrushSize(Number(e.target.value))}
                  className="w-24 accent-purple-600"
                />
                <span className="text-xs text-gray-600 w-7">{brushSize}px</span>
              </div>
              
              <div className="flex items-center gap-2">
                <label className="text-xs font-medium text-gray-700">색상:</label>
                <input
                  type="color"
                  value={brushColor}
                  onChange={(e) => setBrushColor(e.target.value)}
                  className="w-10 h-7 rounded cursor-pointer border border-gray-300"
                />
              </div>
            </div>

            {/* 캔버스 */}
            <div className="mb-3 flex-1 flex items-center justify-center">
              <canvas
                ref={canvasRef}
                width={700}
                height={400}
                onMouseDown={startDraw}
                onMouseMove={draw}
                onMouseUp={endDraw}
                onMouseLeave={endDraw}
                onTouchStart={startDraw}
                onTouchMove={draw}
                onTouchEnd={endDraw}
                className="border-2 border-gray-300 rounded-lg cursor-crosshair bg-white max-w-full max-h-full"
                style={{ width: '700px', height: '400px' }} 
              />
            </div>

            {/* 액션 버튼 */}
            <div className="flex gap-3 mb-3">
              <button
                onClick={clearCanvas}
                className="flex items-center gap-2 px-3 py-2 text-sm bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors shadow-md"
              >
                <Eraser size={18} />
                지우기
              </button>
              <label className="flex items-center gap-2 px-3 py-2 text-sm bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors cursor-pointer shadow-md">
                <Upload size={18} />
                이미지 업로드
                <input
                  type="file"
                  accept="image/*"
                  onChange={uploadImage}
                  className="hidden"
                />
              </label>
            </div>

            {/* 다음 버튼 */}
            <button
              onClick={runInterpretation}
              disabled={!hasAnyDrawing() || isLoading}
              className={`w-full py-3 rounded-lg font-bold text-base transition-all ${
                hasAnyDrawing() && !isLoading
                  ? 'bg-purple-600 text-white hover:bg-purple-700 shadow-lg shadow-purple-500/50'
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed'
              }`}
            >
              {isLoading ? '분석 중...' : '다음'}
            </button>
            
            {!hasAnyDrawing() && (
              <p className="text-center text-red-500 text-xs mt-2">
                최소 한 가지 그림을 그리거나 업로드해주세요
              </p>
            )}
          </div>
        </div>
      </div>
    );
  }

  // 페이지 2 디자인
  return (
    <div className="h-screen bg-gradient-to-br from-blue-50 to-purple-50 p-4 overflow-hidden">
      <div className="max-w-7xl mx-auto h-full flex flex-col">
        {/* 헤더/처음으로 버튼 */}
        <div className="flex items-center justify-between mb-4 border-b pb-2">
          <div className="flex-1 text-center">
            <h1 className="text-3xl font-bold text-purple-600">검사 결과</h1>
          </div>
          <button
            onClick={() => {
              // 초기화 로직
              setCurrentPage(1);
              setMessages([]);
              setInterpretations({ house: "", tree: "", person: "" });
              setCaptions({ house: "", tree: "", person: "" });
              setFinalInterpretation("");
              setQuestionCount(0);
              setIsComplete(false);
              setDrawings({ house: null, tree: null, person: null });
            }}
            className="px-4 py-2 bg-gray-200 text-gray-700 text-sm rounded-lg hover:bg-gray-300 transition-colors shadow-sm"
          >
            처음으로 돌아가기
          </button>
        </div>

        {/* 2단 레이아웃 */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 flex-1 overflow-hidden">
          
          {/* 왼쪽: 그림 요약과 채팅창 */}
          <div className="flex flex-col gap-4 overflow-hidden">
            {/* 그림 요약 */}
            <div className="bg-white rounded-xl shadow-xl p-4">
              <h3 className="text-lg font-bold text-gray-800 mb-3 flex items-center gap-2">
                <Pencil size={18} className="text-purple-600" />
                제출된 그림
              </h3>
              <div className="grid grid-cols-3 gap-3">
                {['house', 'tree', 'person'].map((type) => (
                  <div key={type} className="text-center">
                    <div className="mb-1">
                      <p className="font-semibold text-gray-800 text-xs">{tabNames[type].ko}</p>
                      <p className="text-xs text-gray-500">{tabNames[type].en}</p>
                    </div>
                    <div className="border-2 border-gray-200 rounded-lg p-2 bg-gray-50 h-24 flex items-center justify-center">
                      {drawings[type] ? (
                        <img src={drawings[type]} alt={tabNames[type].ko} className="max-h-full max-w-full object-contain" />
                      ) : (
                        <p className="text-gray-400 text-xs">그림 없음</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* 채팅창 */}
            <div className="bg-white rounded-xl shadow-xl p-4 flex flex-col flex-1 overflow-hidden">
              <h3 className="text-lg font-bold text-gray-800 mb-3">
                {isComplete ? '추가 질문' : `추가 정보 수집 (${questionCount}/5)`}
              </h3>
              
              <div className="flex-1 overflow-y-auto mb-3 space-y-3 p-3 bg-gray-50 rounded-lg shadow-inner">
                {messages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[80%] p-2 rounded-lg text-sm transition-all duration-300 ${
                        msg.role === 'user'
                          ? 'bg-purple-600 text-white rounded-br-none' 
                          : 'bg-white border border-gray-200 rounded-tl-none' 
                      }`}
                    >
                      <p className="whitespace-pre-wrap">{msg.content}</p>
                    </div>
                  </div>
                ))}
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-white border border-gray-200 p-2 rounded-lg text-sm">
                      <p className="text-gray-500">
                        {questionCount >= 5 ? '최종 해석 생성 중...' : '질문 생성 중...'}
                      </p>
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              {/* 입력창 */}
              <div className="flex gap-2">
                <input
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      e.preventDefault();
                      sendMessage();
                    }
                  }}
                  placeholder={isComplete ? "궁금한 점을 물어보세요..." : "답변을 입력해주세요..."}
                  className="flex-1 px-3 py-2 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-600 disabled:bg-gray-100"
                  disabled={isLoading}
                />
                <button
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={sendMessage}
                  disabled={isLoading || !inputMessage.trim()}
                  className="px-4 py-2 text-sm bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors shadow-md"
                >
                  전송
                </button>
              </div>
            </div>
          </div>

          {/* 오른쪽: 개별 해석과 최종 결과 */}
          <div className="flex flex-col gap-4 overflow-y-auto">
            {/* 개별 해석 */}
            <div className="bg-white rounded-xl shadow-xl p-4">
              <h3 className="text-lg font-bold text-gray-800 mb-3 flex items-center gap-2">
                <Brain size={18} className="text-purple-600" />
                개별 해석
              </h3>
              
              {/* 집 해석 */}
              {interpretations.house && (
                <div className="mb-3 border border-blue-200 rounded-lg p-3 bg-blue-50">
                  <h4 className="text-sm font-bold text-blue-800 mb-1 flex items-center gap-2">
                    <MessageCircle className="text-blue-600" size={16} />
                    집 (House)
                  </h4>
                  <p className="text-sm text-gray-600 mb-2">캡션: <span className="italic">{captions.house}</span></p>
                  <div className="bg-white rounded-md p-2 text-xs whitespace-pre-wrap">
                    {interpretations.house}
                  </div>
                </div>
              )}

              {/* 나무 해석 */}
              {interpretations.tree && (
                <div className="mb-3 border border-green-200 rounded-lg p-3 bg-green-50">
                  <h4 className="text-sm font-bold text-green-800 mb-1 flex items-center gap-2">
                    <MessageCircle className="text-green-600" size={16} />
                    나무 (Tree)
                  </h4>
                  <p className="text-sm text-gray-600 mb-2">캡션: <span className="italic">{captions.tree}</span></p>
                  <div className="bg-white rounded-md p-2 text-xs whitespace-pre-wrap">
                    {interpretations.tree}
                  </div>
                </div>
              )}

              {/* 사람 해석 */}
              {interpretations.person && (
                <div className="border border-orange-200 rounded-lg p-3 bg-orange-50">
                  <h4 className="text-sm font-bold text-orange-800 mb-1 flex items-center gap-2">
                    <MessageCircle className="text-orange-600" size={16} />
                    사람 (Person)
                  </h4>
                  <p className="text-sm text-gray-600 mb-2">캡션: <span className="italic">{captions.person}</span></p>
                  <div className="bg-white rounded-md p-2 text-xs whitespace-pre-wrap">
                    {interpretations.person}
                  </div>
                </div>
              )}
            </div>

            {/* 최종 결과 */}
            {finalInterpretation && (
              <div className="bg-white rounded-xl shadow-xl p-4 border-4 border-purple-300">
                <h3 className="text-lg font-bold text-purple-800 mb-3 flex items-center gap-2">
                  <Brain size={20} className="text-purple-600" />
                  최종 종합 해석
                </h3>
                <div className="bg-purple-50 rounded-lg p-3 whitespace-pre-wrap text-sm leading-relaxed">
                  {finalInterpretation}
                </div>
              </div>
            )}
            
            {/* 최종 해석 대기 중 */}
            {!finalInterpretation && questionCount >= 1 && (
                <div className="bg-white rounded-xl shadow-xl p-4 border border-gray-200">
                    <h3 className="text-lg font-bold text-gray-500 mb-3">최종 종합 해석 대기 중...</h3>
                    <p className="text-sm text-gray-400">추가 질문 5회를 모두 완료하면 최종 해석이 나옴.</p>
                </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default HTPChatbot;