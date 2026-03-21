import os
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary

class REFPROP:

    def __init__(self, fluid='PARAHYDROGEN', refprop_path=None):

        # 设置路径
        if refprop_path is None:
            refprop_path = os.environ.get('RPPREFIX', r'C:\Program Files (x86)\REFPROP')
        self.RP = REFPROPFunctionLibrary(refprop_path)
        self.RP.SETPATHdll(refprop_path)

        # 设置工作流体
        err = self.RP.SETFLUIDSdll(fluid + '.FLD')
        if err != 0:
            raise RuntimeError(f"Error setting fluid: {err}")
        
        # 获取单位制枚举值
        self.MASS_BASE_SI = self.RP.GETENUMdll(0, "MASS BASE SI").iEnum

        # 属性代码：名称
        self.prop_names_ordinary = {
            'T': '温度 (K)',
            'P': '压力 (Pa)',
            'D': '密度 (kg/m^3)',
            'H': '焓 (J/kg)',
            'S': '熵 (J/kg-K)',
            'CP': '定压比热 (J/kg-K)',
            'CV': '定容比热 (J/kg-K)',
            'TCX': '导热系数 (W/m-K)',
            'VIS': '黏度 (Pa-s)',
            'TD': '热扩散系数 (m^2/s)',     # 热扩散系数单位未知？其值为cm^2/s单位下数值的百分之一
            'PRANDTL': '普朗特数',
            'V': '比容 (m^3/kg)',           
            'A': '声速 (m/s)',
            'E': '内能 (J/kg)'
        }

        self.prop_names_sat = {
            # 饱和参数
            'Q': '干度',
            'HEATVAPZ': '蒸发热 (J/kg)',
            'STN': '表面张力 (N/m)',
        }

        self.prop_names_crit = {
            # 临界参数
            'TC': '临界温度 (K)',
            'DC': '临界密度 (kg/m^3)',
            'PC': '临界压力 (Pa)'
        }

    def _call(self, in_mode, out_mode, value1, value2, z=[1.0]):

        result = self.RP.REFPROPdll(
            '',                         # 组分字符串（已通过SETFLUIDS设置，无需再次指定）
            in_mode,
            out_mode,
            self.MASS_BASE_SI,
            0,
            0,
            value1, value2,
            z
        )

        if result.ierr != 0:
            raise RuntimeError(f"REFPROPM error: {result.herr}")
        return result.Output
    
    def PRP_SAT(self, vargout, vargin, value, type):
        """
        饱和状态属性计算接口
        vargin: 输入变量类型 ('T' 或 'P')
        type: 相态 ('liquid' 或 'vapor')
        vargout: 输出变量类型
        value: 输入变量数值
        """
        if vargin not in ['T', 'P']:
            raise ValueError("输入变量类型必须为 'T' 或 'P'")
        elif type not in ['liquid', 'vapor']:
            raise ValueError("相态必须为 'liquid' 或 'vapor'")
        elif vargout not in self.prop_names_ordinary and vargout not in self.prop_names_sat:
           raise ValueError(f"输出变量类型 '{vargout}' 不受支持")
        else:
            match vargin, type:
                case 'T', 'liquid':
                    result = self._call('TQ', vargout, value, 0)
                case 'T', 'vapor':
                    result = self._call('TQ', vargout, value, 1)
                case 'P', 'liquid':
                    result = self._call('PQ', vargout, value, 0)
                case 'P', 'vapor':
                    result = self._call('PQ', vargout, value, 1)

        return result[0]
        
    def PRP(self, vargout, vargin1, value1, vargin2, value2):
        """
        非饱和状态属性计算接口
        vargin1, vargin2: 输入变量类型 ('T', 'P', 'D', 'H', 'S')
        value1, value2: 输入变量数值
        vargout: 输出变量类型
        """
        if vargin1 not in ['T', 'P', 'D', 'H', 'S'] or vargin2 not in ['T', 'P', 'D', 'H', 'S']:
            raise ValueError("输入变量类型必须为 'T', 'P', 'D', 'H' 或 'S'")
        elif vargout not in self.prop_names_ordinary:
            raise ValueError(f"输出变量类型 '{vargout}' 不受支持")        
        match (vargin1, vargin2):
            case ('T', 'P'):
                result = self._call('TP', vargout, value1, value2)
            case ('P', 'T'):
                result = self._call('PT', vargout, value1, value2)
            case ('D', 'T'):
                result = self._call('DT', vargout, value1, value2)
            case ('T', 'D'):
                result = self._call('TD', vargout, value1, value2)
            case ('D', 'P'):
                result = self._call('DP', vargout, value1, value2)
            case ('P', 'D'):
                result = self._call('PD', vargout, value1, value2)
            case ('H', 'P'):
                result = self._call('HP', vargout, value1, value2)
            case ('P', 'H'):
                result = self._call('PH', vargout, value1, value2)
            case ('S', 'P'):
                result = self._call('SP', vargout, value1, value2)
            case ('P', 'S'):
                result = self._call('PS', vargout, value1, value2)

        return result[0]

    def PRP_CRIT(self, vargout):
        """
        临界状态属性计算接口
        vargout: 输出变量类型
        """
        if vargout not in self.prop_names_crit:
            raise ValueError(f"输出变量类型 '{vargout}' 不受支持")
        
        result = self._call(' ', vargout, 0, 0)
        return result[0]
    
    def PRP_TRIP(self, vargout):
        """
        三相点属性计算接口
        vargout: 输出变量类型
        """
        if vargout not in ['T_trip', 'P_trip']:
            raise ValueError(f"输出变量类型 '{vargout}' 不受支持")
        
        match vargout:
            case 'T_trip':
                result = self._call(' ', 'TTRP', 0, 0)
            case 'P_trip':
                result = self._call(' ', 'PTRP', 0, 0)
            
        return result[0]

    def PRP_MELT(self, vargin, value):
        """
        熔点属性计算接口
        vargin: 输入变量类型 ('T' 或 'P')
        value: 输入变量数值
        vargout: 输出变量类型
        """
        if vargin not in ['T', 'P']:
            raise ValueError("输入变量类型必须为 'T' 或 'P'")
        
        match vargin:
            case 'T':
                result = self._call('T', 'MELT-TP', value, 0)
                result[0] = result[0] / 1e3
            case 'P':
                result = self._call('P', 'MELT-PT', 0, value)

        return result[0]
        