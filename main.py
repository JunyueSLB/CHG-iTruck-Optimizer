# main.py
from modules.data_preprocessing import data_preprocessing, data_quality_check
from modules.fee_model import calculate_rental_cost, calculate_callout_cost, calculate_special_lane_cost
from modules.rental_to_callout import rental_to_callout
from modules.callout_trip_concat import callout_trip_concat
from modules.callout_to_rental import simulate_callout_to_rental
from modules.rental_calculation import calculate_rental_time_utilization
from modules.callout_predicts import perform_callout_prediction
from modules.callout_predicts import perform_rental_reduction
import pandas as pd
import os

def perform_data_preprocessing(file_path, city_location_path, seg_division_path, special_lane_cost_path, transit_time_path, source_dir, output_dir, rental_truck_list_path, start_date_str=None, end_date_str=None):
    processed_df, city_location, seg_division  = data_preprocessing(file_path, city_location_path, seg_division_path, special_lane_cost_path, rental_truck_list_path, start_date_str, end_date_str)
    processed_df = processed_df[processed_df['SHIPMENT MOT'] != 'TL-RTL']
    msg = data_quality_check(processed_df, city_location, seg_division, None, None, None, transit_time_path)
    os.makedirs(source_dir, exist_ok=True)
    output_file = os.path.join(source_dir, "processed_data.csv")
    processed_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    return output_file, msg

def split_rental_callout(processed_file, rental_truck_list_path, output_dir):
    processed_df = pd.read_csv(processed_file)
    rental_truck_list = pd.read_csv(rental_truck_list_path)
    general_cargo_truck_plates = rental_truck_list[rental_truck_list['Cargo Type'] == 'General Cargo']['Truck Plate'].tolist()
    for shipment_type, file_name in [('Rental', 'rental_info.csv'), ('Callout', 'callout_info.csv')]:
        filtered_df = processed_df[processed_df['Transport Mode'] == shipment_type]
        if shipment_type == 'Rental':
            filtered_df = filtered_df[filtered_df['Truck Plate'].isin(general_cargo_truck_plates)]
        save_path = os.path.join(output_dir, file_name)
        filtered_df.to_csv(save_path, index=False, encoding='utf-8-sig')

def perform_fee_comparison(output_dir, rental_truck_list_path, exchange_rate_path, callout_cost_path, special_lane_cost_path, equipment_vehicle_type_path, rental_cost_path):
    rental_info_path = os.path.join(output_dir, 'rental_info.csv')
    callout_cost_output_path = os.path.join(output_dir, 'callout_cost_output.csv')
    special_lane_cost_output_path = os.path.join(output_dir, 'special_lane_cost_output.csv')
    cost_compare_df, summary_df = rental_to_callout(
        rental_info_path=rental_info_path,
        rental_truck_list_path=rental_truck_list_path,
        exchange_rate_path=exchange_rate_path,
        callout_cost_path=callout_cost_path,
        special_lane_cost_path=special_lane_cost_path,
        equipment_vehicle_type_path=equipment_vehicle_type_path,
        output_path_callout_cost=callout_cost_output_path,
        rental_cost = rental_cost_path,
        output_path_special_lane_cost=special_lane_cost_output_path
    )
    compare_output_path = os.path.join(output_dir, "rental_to_callout_compare.csv")
    cost_compare_df.to_csv(compare_output_path, index=False, encoding='utf-8-sig')
    # summary_df.to_csv(os.path.join(output_dir, 'rental_to_callout_summary.csv'), index=False, encoding='utf-8-sig')
    return compare_output_path


def perform_rental_time_utilization(output_dir, rental_cost_path, callout_cost_path, rental_truck_list_path,
                                    exchange_rate_path):
    rental_info_path = os.path.join(output_dir, "rental_info.csv")
    rental_time_util_output_path = os.path.join(output_dir, "rental_time_utilization_output.csv")
    rental_to_callout_compare_path = os.path.join(output_dir, "rental_to_callout_compare.csv")

    # 传入rental_to_callout_compare_path，函数内部完成计算和合并覆盖
    calculate_rental_time_utilization(
        rental_info_path,
        rental_time_util_output_path,
        rental_cost_path,
        callout_cost_path,
        rental_truck_list_path,
        exchange_rate_path,
        rental_to_callout_compare_path
    )

    # 返回合并后的对比文件路径
    return rental_to_callout_compare_path

def perform_callout_concat(output_dir, transit_time_path):
    callout_info_path = os.path.join(output_dir, "callout_info.csv")
    concat_output_dir = os.path.join(output_dir, "concat results")
    os.makedirs(concat_output_dir, exist_ok=True)
    callout_trip_concat(callout_info_path, transit_time_path, output_dir=concat_output_dir, min_chain_len=10)
    return concat_output_dir

def perform_callout_to_rental(output_dir, rental_cost_path, exchange_rate_path, transit_time_path):
    concat_dir = os.path.join(output_dir, "concat results")
    callout_to_rental_output_path = os.path.join(output_dir, "callout_to_rental cost.csv")
    simulate_callout_to_rental(
        folder_path=concat_dir,
        rental_cost_path=rental_cost_path,
        output_path=callout_to_rental_output_path,
        exchange_rate_path=exchange_rate_path,
        dispatch_time_path=transit_time_path
    )
    return callout_to_rental_output_path

def perform_callout_predict(
    processed_data_path,
    rental_truck_list_path,
    main_lane_forecast_path,
    callout_cost_path,
    special_lane_cost_path,
    exchange_rate_path,
    transit_time_path,
    rental_cost_path,
    Predict_output_dir
):
    """
    从 App 调用：执行 Callout 预测流程
    """
    return perform_callout_prediction(
        processed_data_path=processed_data_path,
        rental_truck_list_path=rental_truck_list_path,
        main_lane_forecast_path=main_lane_forecast_path,
        callout_cost_path=callout_cost_path,
        special_lane_cost_path=special_lane_cost_path,
        exchange_rate_path=exchange_rate_path,
        transit_time_path=transit_time_path,
        rental_cost_path=rental_cost_path,
        Predict_output_dir=Predict_output_dir
    )

def perform_rental_predict(processed_data_path,
                           main_lane_forecast,
                           rental_truck_list_path,
                           exchange_rate_path,
                           callout_cost_path,
                           special_lane_cost_path,
                           equipment_vehicle_type_path,
                           rental_cost,
                           output_dir):
    """
       从 App 调用：执行 Rental 预测流程
       """
    rental_to_callout_compare_path = os.path.join(output_dir, 'future_rental_compare.csv')
    rental_time_util_output_path = os.path.join(output_dir, 'rental_time_utilization_output.csv')
    output_path_callout_cost = os.path.join(output_dir, 'callout_cost_output.csv')
    output_path_special_lane_cost = os.path.join(output_dir, 'special_lane_cost_output.csv')
    df_result, distance_variance_df, compare_df, summary_df = perform_rental_reduction(processed_data_path,
                                   main_lane_forecast,
                                   rental_truck_list_path,
                                   exchange_rate_path,
                                   callout_cost_path,
                                   special_lane_cost_path,
                                   equipment_vehicle_type_path,
                                   output_path_callout_cost,
                                   rental_cost,
                                   output_path_special_lane_cost,
                                   rental_to_callout_compare_path,
                                   rental_time_util_output_path,
                                   output_dir)
    # 保存 compare_df 和 summary_df 到指定路径
    # os.makedirs(output_dir, exist_ok=True)
    # compare_df_path = os.path.join(output_dir, 'future_rental_compare.csv')
    # summary_df_path = os.path.join(output_dir, 'future_rental_summary.csv')

    # compare_df.to_csv(compare_df_path, index=False, encoding='utf-8-sig')
    # summary_df.to_csv(summary_df_path, index=False, encoding='utf-8-sig')

    # print(f"future_rental_compare 保存到：{compare_df_path}")
    # print(f"future_rental_summary 保存到：{summary_df_path}")

if __name__ == "__main__":
    # 保留原main调用逻辑，默认路径硬编码，可供测试
    # 你也可以调用上面封装函数
    source_dir = "data source"
    file_path = "./data source/source data.csv"
    city_location_path = "./data source/city-location.csv"
    seg_division_path = "./data source/segment-division.csv"
    special_lane_cost_path = "./data source/special_lane cost.csv"
    transit_time_path = "./data source/调车时间.csv"
    rental_cost_path = "./data source/rental cost.csv"
    exchange_rate_path = "./data source/exchange rate.xlsx"
    equipment_vehicle_type_path = "./data source/equipment id-vehicle type.csv"
    callout_cost_path = "./data source/callout cost.csv"
    rental_truck_list_path = "./data source/Rental truck list.csv"
    processed_data_path= './results/processed_data.csv'
    main_lane_forecast_path = './data source/Main Lane Operation Forecast.csv'
    Predict_output_dir = "predict results"
    output_dir = "output"
    start_date_str = "2025-01-01"
    end_date_str = "2025-10-30"

    # # 1. 数据预处理及质量检查
    # processed_file, quality_msg = perform_data_preprocessing(
    #     file_path,
    #     city_location_path,
    #     seg_division_path,
    #     special_lane_cost_path,
    #     transit_time_path,
    #     source_dir,
    #     output_dir,
    #     rental_truck_list_path,
    #     start_date_str,
    #     end_date_str
    # )
    #
    # print("数据预处理完成，输出文件:", processed_file)
    # print("数据质量检查信息:", quality_msg)
    #
    # # 2. 租赁与CALL OUT拆分
    # split_rental_callout(processed_file, rental_truck_list_path, output_dir)
    # print("租赁与CALL OUT拆分完成")
    #
    # # 3. 费用模型比较
    # compare_output = perform_fee_comparison(
    #     output_dir,
    #     rental_truck_list_path,
    #     exchange_rate_path,
    #     callout_cost_path,
    #     special_lane_cost_path,
    #     equipment_vehicle_type_path,
    #     rental_cost_path,
    # )
    # print("费用比较完成，输出文件:", compare_output)
    #
    # # 4. 计算租赁时间利用率
    # rental_util_output = perform_rental_time_utilization(output_dir, rental_cost_path, callout_cost_path, rental_truck_list_path, exchange_rate_path)
    # print("租赁时间利用率计算完成，输出文件:", rental_util_output)

    # # 5. CALL OUT行程合并
    # concat_dir = perform_callout_concat(output_dir, transit_time_path)
    # print("CALL OUT行程合并完成，输出目录:", concat_dir)
    #
    # # 6. CALL OUT转租赁模拟
    # callout_to_rental_output = perform_callout_to_rental(
    #     output_dir,
    #     rental_cost_path,
    #     exchange_rate_path,
    #     transit_time_path
    # )
    # print("CALL OUT转租赁模拟完成，输出文件:", callout_to_rental_output)

    # 7. 预测行程
    perform_rental_prediction = perform_rental_predict(
        processed_data_path,
        main_lane_forecast_path,
        rental_truck_list_path,
        exchange_rate_path,
        callout_cost_path,
        special_lane_cost_path,
        equipment_vehicle_type_path,
        rental_cost_path,
        Predict_output_dir
    )

    perform_callout_prediction = perform_callout_predict(
        processed_data_path,
        rental_truck_list_path,
        main_lane_forecast_path,
        callout_cost_path,
        special_lane_cost_path,
        exchange_rate_path,
        transit_time_path,
        rental_cost_path,
        Predict_output_dir)
    pass
