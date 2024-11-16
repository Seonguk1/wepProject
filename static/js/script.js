// User 클래스 정의
class User {
    static instances = [];

    constructor(name) {
      this.name = name;
      this.totalNumSpoken = 0; // 총 말한 횟수
      this.numWordSpoken = 0; // 특정 단어 말한 횟수
    }

    static findUser(name) {
      return User.instances.find(user => user.name === name) || null;
    }

    static addUser(name) {
      const user = new User(name);
      User.instances.push(user);
      return user;
    }
  }
  
  // 유효한 파일 형식인지 확인
  function validFileType(file) {
    const fileTypes = ['text/plain'];
    return fileTypes.includes(file.type);
  }
  const resultDisplay = document.getElementById('resultDisplay');
  // 파일 크기 변환
  function returnFileSize(number) {
    if (number < 1024) {
      return `${number} bytes`;
    } else if (number >= 1024 && number < 1048576) {
      return `${(number / 1024).toFixed(1)} KB`;
    } else if (number >= 1048576) {
      return `${(number / 1048576).toFixed(1)} MB`;
    }
  }

  // 총 말한 횟수 출력
  function totalNumSpoken(){
    // 총 말한 횟수 오름차순 정렬 후 출력
    User.instances.sort((a, b) => b.totalNumSpoken - a.totalNumSpoken);

    let total = 0;

    // 전체 테이블 HTML을 한 번에 생성
    let tableHTML = `<table border="1" style="width:100%; border-collapse: collapse; text-align: center;">
        <thead>
          <tr>
            <th>이름</th>
            <th>총 말한 횟수</th>
          </tr>
        </thead><tbody>
    `;

    // 각 유저 정보 행을 테이블에 추가
    for (let user of User.instances) {
      total += user.totalNumSpoken;
      tableHTML += `
        <tr>
          <td>${user.name}</td>
          <td>${user.totalNumSpoken}회</td>
        </tr>
      `;
    }
    tableHTML += 
    `</tbody><tfoot>
      <tr>
        <th>합계</th>
        <th>${total}회</th>
      `;
    // 테이블 닫기 태그 추가
    tableHTML += `</table>`;

    // 최종적으로 한번에 innerHTML에 삽입
    resultDisplay.innerHTML = tableHTML;
  }
  
  function NumWordSpoken(){
    resultDisplay.innerHTML = "<input id = 'inputWord' type='text' onkeydown='handleKeyDown(event)' placeholder='단어를 입력하세요.'>"
  }

  function handleKeyDown(event){
    const inputWord = document.getElementById("inputWord");
      if (event.key === "Enter") {
        const word = inputWord.value;
        let index = 0;
        while(index<log.length){  
          index = log.indexOf(word,index+word.length);
          if(index == -1)
            break;
          let i = index;
          while(log.slice(--i,i+2)!='\n['){if(i<0)break;}
          if(i<0)continue;
          let userName = '';
          let j;
          for (j = i + 2; log[j] !== ']'; j++) {
            userName += log[j];
          }
          // 대화 형식을 맞추기 위한 예외 처리
          if (!(log.slice(j+1,j+6) == ' [오전 ' || log.slice(j+1,j+6) == ' [오후 ' ))
            continue; 
          User.findUser(userName).numWordSpoken++;
        }
        for (let user of User.instances) {
          if(user.name.toString().includes(word))
            user.numWordSpoken -= user.totalNumSpoken;
        }
        User.instances.sort((a, b) => b.numWordSpoken - a.numWordSpoken);
        let total=0;
        let tableHTML = `<table border="1" style="width:100%; border-collapse: collapse; text-align: center;">
            <thead>
              <tr>
                <th>이름</th>
                <th>"${word}" 말한 횟수</th>
              </tr>
            </thead><tbody>
        `;
        for (let user of User.instances) {
          total += user.numWordSpoken;
          tableHTML += `
            <tr>
              <td>${user.name}</td>
              <td>${user.numWordSpoken}회</td>
            </tr>
          `;
          user.numWordSpoken=0;
        }
        tableHTML += 
        `</tbody><tfoot>
          <tr>
            <th>합계</th>
            <th>${total}회</th>
          `;
        resultDisplay.innerHTML = tableHTML;

      }
    }

    
    
    // let reader, log;
    // input.addEventListener('change', () => {
    //   fileInfo.textContent = ""; // 파일 정보 초기화
    //   resultDisplay.textContent = "분석 결과가 여기에 표시됩니다."; // 결과 초기화
    //   User.instances = []; // User 인스턴스 초기화

    //   const selectedFiles = input.files;
    //   for (const file of selectedFiles) {
    //     if (validFileType(file)) {
    //       fileInfo.textContent = `파일명: ${file.name}, 파일 크기: ${returnFileSize(file.size)}.`;

    //       reader = new FileReader();
    //       reader.onload = function () {
    //         log = reader.result;
            
    //         for (let i = 0; i < log.length; i++) {
    //           //대화 이용자 확인
    //           if (log[i] === '\n'){
    //             if(log[i + 1] === '[') {
    //               let userName = '';
    //               let j;
    //               for (j = i + 2; log[j] !== ']'; j++) {
    //                 userName += log[j];
    //               }
                  
    //               // 대화 형식을 맞추기 위한 예외 처리
    //               if (!(log.slice(j+1,j+6) == ' [오전 ' || log.slice(j+1,j+6) == ' [오후 ' ))
    //                 continue;
                  

    //               let user = User.findUser(userName);
    //               if (user) {
    //                 user.totalNumSpoken++;
    //               } else {
    //                 user = User.addUser(userName);
    //                 user.totalNumSpoken++;
    //               }
    //             }
    //           }
    //             // 대화 날짜 확인
    //             else if(log.slice(i+1,i+17)=='--------------- '){
    //               let j=i+17;
    //               let date = ''; 
    //               while(log[j]!='-'){
    //                 date+=log[j];
    //                 j++
    //               }
    //               date = date.trim(); 
    //             }
              
    //           else{

    //           }
    //         }

            
    //       };
    //       reader.readAsText(file, "UTF-8");
    //     } else {
    //       fileInfo.textContent = `파일명 ${file.name}: .txt 파일을 선택하세요.`;
    //     }
    //   }
    // });

    
