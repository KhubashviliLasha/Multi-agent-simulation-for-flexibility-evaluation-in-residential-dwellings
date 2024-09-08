# it calculate total energy consumptions acording my given data from internet and not real data from iris 
import pandas as pd
import os
import ipywidgets as widgets
from IPython.display import display, HTML
import plotly.graph_objs as go

class DeviceConsumptionCalculator:
    def __init__(self, csv_directory):
        self.csv_directory = csv_directory
        self.house_devices = self.get_unique_devices()
        self.consumption_data = None

    def get_unique_devices(self):
        house_devices = {}
        for filename in os.listdir(self.csv_directory):
            if filename.endswith('.csv'):
                house_id = filename.split('.')[0]
                file_path = os.path.join(self.csv_directory, filename)
                df = pd.read_csv(file_path)
                devices = df['Name'].str.split('(').str[0].str.strip().unique()
                house_devices[house_id] = sorted(devices)
        return house_devices

    def calculate_device_consumption(self, house_ids, devices):
        consumption_data = {}
        for filename in os.listdir(self.csv_directory):
            if filename.endswith('.csv'):
                house_id = filename.split('.')[0]
                if house_ids != ['All'] and house_id not in house_ids:
                    continue
                file_path = os.path.join(self.csv_directory, filename)
                df = pd.read_csv(file_path)
                house_consumption = {}
                for device in devices:
                    device_row = df[df['Name'].str.lower().str.contains(device.lower(), na=False)]
                    if not device_row.empty:
                        annual_energy = device_row['typical annual energy [kwh/yr]'].sum()
                        house_consumption[device] = annual_energy
                    else:
                        house_consumption[device] = 0
                consumption_data[house_id] = house_consumption
        return consumption_data

    def on_house_selection_change(self, change):
        selected_houses = change['new']
        if 'All' in selected_houses:
            devices = set()
            for house_devices in self.house_devices.values():
                devices.update(house_devices)
        else:
            devices = set()
            for house in selected_houses:
                devices.update(self.house_devices.get(house, []))
        self.device_selector.options = sorted(devices)

    def on_button_click(self, b):
        selected_houses = [str(house_id) for house_id in self.house_selector.value]
        selected_devices = self.device_selector.value

        if not selected_houses:
            print("Please select at least one house ID")
            return
        if not selected_devices:
            print("Please select at least one device")
            return

        self.consumption_data = self.calculate_device_consumption(selected_houses, selected_devices)
        self.display_results()
        self.plot_total_consumption()

    def create_widgets(self):
        house_ids = ['All'] + list(self.house_devices.keys())
        all_devices = set()
        for devices in self.house_devices.values():
            all_devices.update(devices)

        self.house_selector = widgets.SelectMultiple(
            options=house_ids,
            description='Select Houses:',
            disabled=False
        )

        self.device_selector = widgets.SelectMultiple(
            options=sorted(all_devices),
            description='Select Devices:',
            disabled=False
        )

        self.house_selector.observe(self.on_house_selection_change, names='value')

        self.button = widgets.Button(description="Calculate Consumption")
        self.button.on_click(self.on_button_click)

        display(self.house_selector, self.device_selector, self.button)

    def display_results(self):
        if self.consumption_data is None:
            print("No data to display. Please calculate consumption first.")
            return

        df = pd.DataFrame(self.consumption_data).T.fillna(0)
        df['Total'] = df.sum(axis=1)
        df = df.sort_values('Total', ascending=False)

        display(HTML("<h3>Consumption Data</h3>"))
        display(df)

        total_consumption = df['Total'].sum()
        print(f"\nTotal Consumption: {total_consumption:.2f} kWh/year")

    def plot_total_consumption(self):
        if self.consumption_data:
            df = pd.DataFrame(self.consumption_data).T
            total_consumption_by_appliance = df.sum().sort_values(ascending=False)

            fig = go.Figure(go.Bar(
                x=total_consumption_by_appliance.index,
                y=total_consumption_by_appliance.values,
                marker_color='rgb(55, 83, 109)'
            ))

            fig.update_layout(
                title='Total Consumption by Appliance',
                xaxis_title='Appliance',
                yaxis_title='Total Consumption (kWh/year)',
                xaxis_tickangle=-45,
                margin=dict(l=50, r=50, t=50, b=100)
            )

            display(HTML("<h3>Total Consumption by Appliance</h3>"))
            display(total_consumption_by_appliance)
            fig.show()
        else:
            print("No data was calculated.")


# this code is printing every details about all houses and appliances and shows graph acording chosen appliances 

# import pandas as pd
# from collections import defaultdict
# import re
# import matplotlib.pyplot as plt
# import tkinter as tk
# from tkinter import ttk

# class ApplianceConsumptionCalculatorIrise:
#     def __init__(self, irise):
#         self.irise = irise
#         self.appliance_consumption = defaultdict(lambda: defaultdict(float))
#         self.appliance_count = defaultdict(int)
#         self.houses_with_appliances = defaultdict(int)
#         self.total_houses = 0
#         self.house_ids = []
#         self.devices = []

#     def normalize_appliance_name(self, name):
#         name = name.split('(')[0].strip()
#         normalized_name = name.replace("_", "").lower()
#         normalized_name = normalized_name.replace(" ", "")
#         normalized_name = re.sub(r'[^a-zA-Z]', '', name.lower())
#         normalized_name = ''.join([i for i in normalized_name if not i.isdigit()])
#         return normalized_name

#     def calculate_consumption(self):
#         for house in self.irise.get_houses():
#             self.total_houses += 1
#             house_id = house.id
#             self.house_ids.append(house_id)
#             house_appliances = set()
#             print(f"\nChecking house ID:{house_id}")

#             for appliance in house.get_appliances():
#                 normalized_name = self.normalize_appliance_name(appliance.name)
#                 if normalized_name not in self.devices:
#                     self.devices.append(normalized_name)
#                     print(f"  Appliance: {appliance.name} (Normalized: {normalized_name})")
                
#                 consumptions = appliance.get_consumptions_kWh()
#                 total_consumption = sum(consumptions)
                
#                 self.appliance_consumption[normalized_name][house_id] += total_consumption
#                 self.appliance_count[normalized_name] += 1
#                 house_appliances.add(normalized_name)

#                 print(f"    Consumption: {total_consumption:.2f} kWh")

#             for appliance in house_appliances:
#                 self.houses_with_appliances[appliance] += 1

#         print(f"\nTotal houses checked: {self.total_houses}")
#         for appliance, count in self.houses_with_appliances.items():
#             print(f"houses with {appliance}: {count}")
        
        
#     print("\nAppliance Consumption Summary:")
#     def get_consumption_summary(self, selected_houses, selected_devices):
#         summary = []
#         for appliance in selected_devices:
#             house_consumptions = self.appliance_consumption[appliance]
#             total_appliance_consumption = sum(house_consumptions[house] for house in selected_houses if house in house_consumptions)
#             avg_consumption = total_appliance_consumption / self.appliance_count[appliance] if self.appliance_count[appliance] > 0 else 0
#             presence_count = sum(1 for house in selected_houses if house in house_consumptions)
#             presence_percentage = (presence_count / len(selected_houses)) * 100 if selected_houses else 0
            
#             summary.append({
#                 'Appliance': appliance.capitalize(),
#                 'Total Consumption (kWh/year)': total_appliance_consumption,
#                 'Average Consumption per Appliance (kWh/year)': avg_consumption,
#                 'Presence in Selected Houses (%)': presence_percentage
#             })
#             print(f"\n{appliance.capitalize()}:")
#             print(f"  Total Consumption: {total_appliance_consumption:.2f} kWh/year")
#             print(f"  Average Consumption per Appliance: {avg_consumption:.2f} kWh/year")
#             print(f"  Present in {self.houses_with_appliances[appliance]} out of {self.total_houses} houses ({presence_percentage:.2f}%)")


#         df_consumption = pd.DataFrame(self.appliance_consumption)
#         df_consumption.index.name = 'House ID'
#         print("\nConsumption DataFrame:")
#         print(df_consumption)


#         total_consumption_by_appliance = df_consumption.sum()
#         print("\nTotal Consumption by Appliance:")
#         print(total_consumption_by_appliance)

#         return pd.DataFrame(summary)
    
    

#     def plot_total_consumption_by_appliance(self, selected_houses, selected_devices):
#         summary_df = self.get_consumption_summary(selected_houses, selected_devices)
        
#         plt.figure(figsize=(12, 6))
#         summary_df.plot(x='Appliance', y='Total Consumption (kWh/year)', kind='bar')
#         plt.title('Total Consumption by Appliance')
#         plt.xlabel('Appliance')
#         plt.ylabel('Total Consumption (kWh/year)')
#         plt.xticks(rotation=45, ha='right')
#         plt.tight_layout()
#         plt.show()


#     def center_window(self, window,width,height):
#         screen_width = window.winfo_screenwidth()
#         screen_height = window.winfo_screenheight()
#         x = (screen_width - width) // 2
#         y = (screen_height - height) // 2
#         window.geometry(f'{width}x{height}+{x}+{y}')

#     def run_gui(self):
#         self.calculate_consumption()

#         root = tk.Tk()
#         root.title("Appliance Consumption Calculator")

#         # Set window size and center it
#         window_width = 600
#         window_height = 400
#         self.center_window(root, window_width, window_height)

#         frame = ttk.Frame(root, padding="10")
#         frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
#         root.columnconfigure(0, weight=1)
#         root.rowconfigure(0, weight=1)

#         house_label = ttk.Label(frame, text="Select Houses:")
#         house_label.grid(row=0, column=0, sticky=tk.W)
#         house_listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, exportselection=0)
#         house_listbox.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
#         for house_id in self.house_ids:
#             house_listbox.insert(tk.END, house_id)

#         device_label = ttk.Label(frame, text="Select Devices:")
#         device_label.grid(row=0, column=1, sticky=tk.W)
#         device_listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, exportselection=0)
#         device_listbox.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
#         for device in self.devices:
#             device_listbox.insert(tk.END, device)

#         frame.columnconfigure(0, weight=1)
#         frame.columnconfigure(1, weight=1)
#         frame.rowconfigure(1, weight=1)

#         def on_calculate():
#             selected_houses = [house_listbox.get(i) for i in house_listbox.curselection()]
#             selected_devices = [device_listbox.get(i) for i in device_listbox.curselection()]
#             if not selected_houses or not selected_devices:
#                 tk.messagebox.showerror("Error", "Please select at least one house and one device.")
#                 return
#             summary_df = self.get_consumption_summary(selected_houses, selected_devices)
#             print(summary_df)
#             self.plot_total_consumption_by_appliance(selected_houses, selected_devices)

#         calculate_button = ttk.Button(frame, text="Calculate and Plot", command=on_calculate)
#         calculate_button.grid(row=2, column=0, columnspan=2, pady=10)

#         root.mainloop()

# Example usage:
# irise = ...  # Your irise object
# calculator = ApplianceConsumptionCalculatorIrise(irise)
# calculator.run_gui()

# avg_consumption = total_appliance_consumption / self.appliance_count[appliance] if self.appliance_count[appliance] > 0 else 0
# total_appliance_consumption = sum(house_consumptions[house] for house in selected_houses if house in house_consumptions)




# it calculates consumptions for 1 year for each appliances and ids (you can choose)

import pandas as pd
from collections import defaultdict
import re
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox

class ApplianceConsumptionCalculatorIrise:
    def __init__(self, irise):
        self.irise = irise
        self.appliance_consumption = defaultdict(lambda: defaultdict(float))
        self.appliance_count = defaultdict(int)
        self.houses_with_appliances = defaultdict(int)
        self.total_houses = 0
        self.house_ids = []
        self.devices = []
        self.house_appliances_map = defaultdict(set)

    def normalize_appliance_name(self, name):
        name = name.split('(')[0].strip()
        normalized_name = name.replace("_", "").lower()
        normalized_name = normalized_name.replace(" ", "")
        normalized_name = re.sub(r'[^a-zA-Z]', '', name.lower())
        normalized_name = ''.join([i for i in normalized_name if not i.isdigit()])
        return normalized_name

    def calculate_consumption(self):
        for house in self.irise.get_houses():
            self.total_houses += 1
            house_id = house.id
            self.house_ids.append(house_id)
            house_appliances = set()

            for appliance in house.get_appliances():
                normalized_name = self.normalize_appliance_name(appliance.name)
                if normalized_name not in self.devices:
                    self.devices.append(normalized_name)
                
                consumptions = appliance.get_consumptions_kWh()
                total_consumption = sum(consumptions)
                
                self.appliance_consumption[normalized_name][house_id] += total_consumption
                self.appliance_count[normalized_name] += 1
                house_appliances.add(normalized_name)

            self.house_appliances_map[house_id] = house_appliances

            for appliance in house_appliances:
                self.houses_with_appliances[appliance] += 1

    def get_consumption_summary(self, selected_houses, selected_devices):
        summary = []
        missing_appliances = defaultdict(list)
        for appliance in selected_devices:
            house_consumptions = self.appliance_consumption[appliance]
            total_appliance_consumption = sum(house_consumptions[house] for house in selected_houses if house in house_consumptions)
            avg_consumption = total_appliance_consumption / self.appliance_count[appliance] if self.appliance_count[appliance] > 0 else 0
            presence_count = sum(1 for house in selected_houses if house in house_consumptions)
            presence_percentage = (presence_count / len(selected_houses)) * 100 if selected_houses else 0
            
            for house in selected_houses:
                if house not in house_consumptions:
                    missing_appliances[appliance].append(house)
            
            summary.append({
                'Appliance': appliance.capitalize(),
                'Total Consumption (kWh/year)': total_appliance_consumption,
                'Average Consumption per Appliance (kWh/year)': avg_consumption,
                'Presence in Selected Houses (%)': presence_percentage
            })
        return pd.DataFrame(summary), missing_appliances

    def plot_total_consumption_by_appliance(self, selected_houses, selected_devices, missing_appliances):
        summary_df, _ = self.get_consumption_summary(selected_houses, selected_devices)
        
        plt.close('all')
        fig, ax = plt.subplots(figsize=(12, 6))
        
        summary_df.plot(x='Appliance', y='Total Consumption (kWh/year)', kind='bar', ax=ax)
        title = f'Total Consumption by Appliance\nSelected Houses: {", ".join(map(str, selected_houses))}'
        if missing_appliances:
            title += '\nSome appliances are missing in certain houses'
        plt.title(title, fontsize=10)
        plt.xlabel('Appliance')
        plt.ylabel('Total Consumption (kWh/year)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def center_window(self, window, width, height):
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        window.geometry(f'{width}x{height}+{x}+{y}')

    def update_device_listbox(self, house_listbox, device_listbox):
        selected_houses = [house_listbox.get(i) for i in house_listbox.curselection()]
        if not selected_houses:
            selected_houses = self.house_ids
            
        appliances = set()
        for house_id in selected_houses:
            appliances.update(self.house_appliances_map[house_id])

        device_listbox.delete(0, tk.END)
        for appliance in sorted(appliances):
            device_listbox.insert(tk.END, appliance)

    def run_gui(self):
        self.calculate_consumption()

        root = tk.Tk()
        root.title("Appliance Consumption Calculator")

        window_width = 600
        window_height = 400
        self.center_window(root, window_width, window_height)

        frame = ttk.Frame(root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        house_label = ttk.Label(frame, text="Select Houses:")
        house_label.grid(row=0, column=0, sticky=tk.W)
        house_listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, exportselection=0)
        house_listbox.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        for house_id in self.house_ids:
            house_listbox.insert(tk.END, house_id)

        device_label = ttk.Label(frame, text="Select Devices:")
        device_label.grid(row=0, column=1, sticky=tk.W)
        device_listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, exportselection=0)
        device_listbox.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(1, weight=1)

        def on_calculate():
            selected_houses = [house_listbox.get(i) for i in house_listbox.curselection()]
            selected_devices = [device_listbox.get(i) for i in device_listbox.curselection()]
            if not selected_houses or not selected_devices:
                messagebox.showerror("Error", "Please select at least one house and one device.")
                return
            
            print(f"Selected Houses: {selected_houses}")
            print(f"Selected Appliances: {selected_devices}")

            summary_df, missing_appliances = self.get_consumption_summary(selected_houses, selected_devices)
            print(summary_df)
            
            if missing_appliances:
                print("Missing Appliances in Houses:")
                missing_info = []
                for appliance, houses in missing_appliances.items():
                    if houses:
                        missing_info.append(f"{appliance.capitalize()} is not present in houses: {', '.join(map(str, houses))}")
                        print(f"House(s) {', '.join(map(str, houses))} do not have {appliance}")
                
                messagebox.showinfo("Missing Appliances", "\n".join(missing_info))
            
            self.plot_total_consumption_by_appliance(selected_houses, selected_devices, missing_appliances)

        calculate_button = ttk.Button(frame, text="Calculate and Plot", command=on_calculate)
        calculate_button.grid(row=2, column=0, columnspan=2, pady=10)

        house_listbox.bind('<<ListboxSelect>>', lambda event: self.update_device_listbox(house_listbox, device_listbox))

        root.mainloop()



# this code shows appliances consumptions one by one for each houses we can filter acording seasons of year or week days. 
import pandas as pd
import plotly.graph_objs as go
from collections import defaultdict
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import calendar
from datetime import datetime, timedelta

class DeviceConsumption24_separate:
    def __init__(self, irise):
        self.irise = irise
        self.house_selector = None
        self.weekday_selector = None
        self.season_selector = None
        self.button = None
        self.start_date = self.irise.datetimes[0]
        self.end_date = self.irise.datetimes[-1]
        self.year = self.start_date.year
        self.days_in_year = 366 if self.is_leap_year(self.year) else 365

    def is_leap_year(self, year):
        return calendar.isleap(year)

    def filter_by_weekday(self, date, selected_weekdays):
        weekday = calendar.day_name[date.weekday()]
        return weekday in selected_weekdays

    def filter_by_season(self, date, selected_seasons):
        seasons = {
            'Winter': (12, 1, 2),
            'Spring': (3, 4, 5),
            'Summer': (6, 7, 8),
            'Autumn': (9, 10, 11)
        }
        month = date.month
        for season, months in seasons.items():
            if month in months and season in selected_seasons:
                return True
        return False

    def calculate_appliance_consumption(self, house_id, selected_weekdays, selected_seasons):
        appliance_consumption = {}
        
        house = self.irise.get_house(house_id)
        print(f"\nAnalyzing House ID: {house_id}")
        
        for appliance in house.get_appliances():
            appliance_id = appliance.appliance_id
            appliance_name = str(appliance).split(":")[1].strip().split(" ")[0]
            
            consumptions = np.array(appliance.get_consumptions_kWh()) / self.days_in_year
            total_consumption = sum(consumptions)
            
            hourly_data = defaultdict(float)
            for date, consumption in zip(self.irise.datetimes, consumptions):
                hour = date.hour
                
                if self.filter_by_weekday(date, selected_weekdays) and self.filter_by_season(date, selected_seasons):
                    hourly_data[hour] += consumption
            
            appliance_consumption[appliance_id] = {
                'name': appliance_name,
                'total_consumption': total_consumption,
                'hourly_consumption': dict(hourly_data)
            }
            
            print(f"  Appliance ID {appliance_id}: {appliance_name}")
            print(f"    Total Consumption: {total_consumption:.2f} kWh")
        
        return appliance_consumption

    def plot_consumption(self, selected_houses, consumption_data):
        hours = ['%ih' % i for i in range(24)]
        fig = go.Figure()

        for house_id, house_data in consumption_data.items():
            for appliance_id, data in house_data.items():
                hourly_consumption_list = [data['hourly_consumption'].get(i, 0) for i in range(24)]
                fig.add_trace(go.Bar(
                    name=f"House {house_id} - {data['name']} (ID: {appliance_id})",
                    x=hours,
                    y=hourly_consumption_list
                ))

        fig.update_layout(
            title=f'Hourly Consumption by Appliance (Houses: {", ".join(map(str, selected_houses))})',
            xaxis_title='Hour of the Day',
            yaxis_title='Consumption (kWh)',
            barmode='group'
        )

        fig.show()

        fig_total = go.Figure()

        for house_id, house_data in consumption_data.items():
            fig_total.add_trace(go.Bar(
                name=f'House {house_id}',
                x=[f"{data['name']} (ID: {appliance_id})" for appliance_id, data in house_data.items()],
                y=[data['total_consumption'] for data in house_data.values()]
            ))

        fig_total.update_layout(
            title=f'Annual Consumption by Appliance (Houses: {", ".join(map(str, selected_houses))})',
            xaxis_title='Appliance',
            yaxis_title='Total Consumption (kWh)',
            xaxis_tickangle=-45,
            barmode='group'
        )

        fig_total.show()

    def analyze_houses(self, selected_houses, selected_weekdays, selected_seasons):
        if not selected_houses:
            print("Please select at least one house.")
            return
        
        consumption_data = {}
        for house_id in selected_houses:
            try:
                consumption_data[house_id] = self.calculate_appliance_consumption(house_id, selected_weekdays, selected_seasons)
            except Exception as e:
                print(f"An error occurred for House ID {house_id}: {str(e)}")
        
        if consumption_data:
            self.plot_consumption(selected_houses, consumption_data)

    def plot_total_consumption(self, selected_houses, selected_weekdays, selected_seasons):
        global_hourly_consumptions_kWh = self.irise.get_av_hourly_consumptions_kWh()
        
        # Calculate the average daily consumption correctly
        days = len(global_hourly_consumptions_kWh) // 24
        daily_consumption = [sum(global_hourly_consumptions_kWh[i::24]) / days for i in range(24)]
        
        fig = go.Figure()

        # Add trace for global hourly consumption
        fig.add_trace(go.Scatter(
            x=list(range(24)),
            y=daily_consumption,
            name='Global Average Annual Consumption',
            line=dict(color='black', width=2)
        ))

        # Add traces for selected houses
        for house_id in selected_houses:
            house = self.irise.get_house(house_id)
            house_consumptions = house.get_consumptions_kWh()
            daily_house_consumption = np.array([sum(house_consumptions[i::24]) / days for i in range(24)])/self.days_in_year
            fig.add_trace(go.Scatter(
                x=list(range(24)),
                y=daily_house_consumption,
                name=f'House {house_id} Average Annual Consumption For All APP',
                line=dict(dash='dash')
            ))

        fig.update_layout(
            title='Average Annual Consumption Comparison',
            xaxis_title='Hour of the Day',
            yaxis_title='Consumption (kWh)',
            xaxis=dict(tickmode='linear', tick0=0, dtick=1)
        )
        
        fig.show()

    def on_button_click(self, b):
        selected_houses = [int(house_id) for house_id in self.house_selector.value]
        selected_weekdays = list(self.weekday_selector.value)
        selected_seasons = list(self.season_selector.value)
        self.analyze_houses(selected_houses, selected_weekdays, selected_seasons)
        self.plot_total_consumption(selected_houses, selected_weekdays, selected_seasons)

    def create_widgets(self, house_ids):
        self.house_selector = widgets.SelectMultiple(
            options=house_ids,
            description='Select Houses:',
            disabled=False
        )

        self.weekday_selector = widgets.SelectMultiple(
            options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            description='Select Weekdays:',
            disabled=False
        )

        self.season_selector = widgets.SelectMultiple(
            options=['Winter', 'Spring', 'Summer', 'Autumn'],
            description='Select Seasons:',
            disabled=False
        )

        self.button = widgets.Button(description="Generate Plots")
        self.button.on_click(self.on_button_click)

        display(self.house_selector, self.weekday_selector, self.season_selector, self.button)


# this code does same as up but with normalize names, it shows same appliances together for all houses 
import pandas as pd
import plotly.graph_objs as go
from collections import defaultdict
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import calendar
from datetime import datetime, timedelta
import re

class DeviceConsumption24:
    def __init__(self, irise):
        self.irise = irise
        self.house_selector = None
        self.weekday_selector = None
        self.season_selector = None
        self.button = None
        self.start_date = self.irise.datetimes[0]
        self.end_date = self.irise.datetimes[-1]
        self.year = self.start_date.year
        self.days_in_year = 366 if self.is_leap_year(self.year) else 365

    def is_leap_year(self, year):
        return calendar.isleap(year)

    def filter_by_weekday(self, date, selected_weekdays):
        weekday = calendar.day_name[date.weekday()]
        return weekday in selected_weekdays

    def filter_by_season(self, date, selected_seasons):
        seasons = {
            'Winter': (12, 1, 2),
            'Spring': (3, 4, 5),
            'Summer': (6, 7, 8),
            'Autumn': (9, 10, 11)
        }
        month = date.month
        for season, months in seasons.items():
            if month in months and season in selected_seasons:
                return True
        return False

    def normalize_appliance_name(self, name):
        name = name.split('(')[0].strip()
        normalized_name = re.sub(r'[^a-zA-Z]', 
                                 ' ', 
                                 name.lower())
        return normalized_name

    def calculate_appliance_consumption(self, house_id, selected_weekdays, selected_seasons):
        appliance_consumption = defaultdict(lambda: {'total_consumption': 0, 'hourly_consumption': defaultdict(float)})
        
        house = self.irise.get_house(house_id)
        print(f"\nAnalyzing House ID: {house_id}")
        
        for appliance in house.get_appliances():
            appliance_name = str(appliance).split(":")[1].strip().split(" ")[0]
            normalized_name = self.normalize_appliance_name(appliance_name)
            
            consumptions = np.array(appliance.get_consumptions_kWh()) / self.days_in_year
            total_consumption = sum(consumptions)
            
            for date, consumption in zip(self.irise.datetimes, consumptions):
                hour = date.hour
                if self.filter_by_weekday(date, selected_weekdays) and self.filter_by_season(date, selected_seasons):
                    appliance_consumption[normalized_name]['hourly_consumption'][hour] += consumption
            
            appliance_consumption[normalized_name]['total_consumption'] += total_consumption
            print(f"  Appliance: {appliance_name} (Normalized: {normalized_name})")
            print(f"    Total Consumption: {total_consumption:.2f} kWh")
        
        return dict(appliance_consumption)

    def analyze_houses(self, selected_houses, selected_weekdays, selected_seasons):
        if not selected_houses:
            print("Please select at least one house.")
            return
        
        aggregated_consumption = defaultdict(lambda: {'total_consumption': 0, 'hourly_consumption': defaultdict(float)})
        
        for house_id in selected_houses:
            try:
                house_consumption = self.calculate_appliance_consumption(house_id, selected_weekdays, selected_seasons)
                for appliance, data in house_consumption.items():
                    aggregated_consumption[appliance]['total_consumption'] += data['total_consumption']
                    for hour, consumption in data['hourly_consumption'].items():
                        aggregated_consumption[appliance]['hourly_consumption'][hour] += consumption
            except Exception as e:
                print(f"An error occurred for House ID {house_id}: {str(e)}")
        
        if aggregated_consumption:
            self.plot_consumption(selected_houses, dict(aggregated_consumption))

    def plot_consumption(self, selected_houses, consumption_data):
        hours = ['%ih' % i for i in range(24)]
        fig = go.Figure()

        for appliance, data in consumption_data.items():
            hourly_consumption_list = [data['hourly_consumption'].get(i, 0) for i in range(24)]
            fig.add_trace(go.Bar(
                name=f"{appliance}",
                x=hours,
                y=hourly_consumption_list
            ))

        fig.update_layout(
            title=f'Hourly Consumption by Appliance (Houses: {", ".join(map(str, selected_houses))})',
            xaxis_title='Hour of the Day',
            yaxis_title='Consumption (kWh)',
            barmode='group'
        )

        fig.show()

        fig_total = go.Figure()

        fig_total.add_trace(go.Bar(
            name='Total Consumption',
            x=list(consumption_data.keys()),
            y=[data['total_consumption'] for data in consumption_data.values()]
        ))

        fig_total.update_layout(
            title=f'Annual Consumption by Appliance (Houses: {", ".join(map(str, selected_houses))})',
            xaxis_title='Appliance',
            yaxis_title='Total Consumption (kWh)',
            xaxis_tickangle=-45,
            barmode='group'
        )

        fig_total.show()

    def plot_total_consumption(self, selected_houses, selected_weekdays, selected_seasons):
        global_hourly_consumptions_kWh = self.irise.get_av_hourly_consumptions_kWh()
        
        days = len(global_hourly_consumptions_kWh) // 24
        daily_consumption = [sum(global_hourly_consumptions_kWh[i::24]) / days for i in range(24)]
        
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=list(range(24)),
            y=daily_consumption,
            name='Global Average Annual Consumption',
            line=dict(color='black', width=2)
        ))

        for house_id in selected_houses:
            house = self.irise.get_house(house_id)
            house_consumptions = house.get_consumptions_kWh()
            daily_house_consumption = np.array([sum(house_consumptions[i::24]) / days for i in range(24)]) / self.days_in_year
            fig.add_trace(go.Scatter(
                x=list(range(24)),
                y=daily_house_consumption,
                name=f'House {house_id} Average Annual Consumption For All APP',
                line=dict(dash='dash')
            ))

        fig.update_layout(
            title='Average Annual Consumption Comparison',
            xaxis_title='Hour of the Day',
            yaxis_title='Consumption (kWh)',
            xaxis=dict(tickmode='linear', tick0=0, dtick=1)
        )
        
        fig.show()

    def on_button_click(self, b):
        selected_houses = [int(house_id) for house_id in self.house_selector.value]
        selected_weekdays = list(self.weekday_selector.value)
        selected_seasons = list(self.season_selector.value)
        self.analyze_houses(selected_houses, selected_weekdays, selected_seasons)
        self.plot_total_consumption(selected_houses, selected_weekdays, selected_seasons)

    def create_widgets(self, house_ids):
        self.house_selector = widgets.SelectMultiple(
            options=house_ids,
            description='Select Houses:',
            disabled=False
        )

        self.weekday_selector = widgets.SelectMultiple(
            options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            description='Select Weekdays:',
            disabled=False
        )

        self.season_selector = widgets.SelectMultiple(
            options=['Winter', 'Spring', 'Summer', 'Autumn'],
            description='Select Seasons:',
            disabled=False
        )

        self.button = widgets.Button(description="Generate Plots")
        self.button.on_click(self.on_button_click)

        display(self.house_selector, self.weekday_selector, self.season_selector, self.button)




# this code does same but with line diagram and for all year without devition of seasons or weak days. 
import pandas as pd
import plotly.graph_objs as go
from collections import defaultdict

class DeviceConsumption24h:
    def __init__(self, irise):
        self.irise = irise

    def calculate_appliance_consumption(self, house_id):
        appliance_consumption = {}
        
        house = self.irise.get_house(house_id)
        print(f"\nAnalyzing House ID: {house_id}")
        
        for appliance in house.get_appliances():
            appliance_id = appliance.appliance_id
            appliance_name = str(appliance).split(":")[1].strip().split(" ")[0]
            
            consumptions = appliance.get_consumptions_kWh()
            total_consumption = sum(consumptions)
            
            hourly_data = defaultdict(float)
            for i, consumption in enumerate(consumptions):
                hour = i % 24
                hourly_data[hour] += consumption
            
            appliance_consumption[appliance_id] = {
                'name': appliance_name,
                'total_consumption': total_consumption,
                'hourly_consumption': dict(hourly_data)
            }
            
            print(f"  Appliance ID {appliance_id}: {appliance_name}")
            print(f"    Total Consumption: {total_consumption:.2f} kWh")
        
        return appliance_consumption

    def plot_consumption(self, selected_houses, consumption_data):
        hours = list(range(24))
        fig = go.Figure()

        # Add global hourly consumption
        global_hourly_consumptions_kWh = self.irise.get_av_hourly_consumptions_kWh()
        days = len(global_hourly_consumptions_kWh) // 24
        daily_consumption = [sum(global_hourly_consumptions_kWh[i::24]) / days for i in range(24)]
        fig.add_trace(go.Scatter(
            x=hours,
            y=daily_consumption,
            name='Global Average Daily Consumption',
            line=dict(color='black', width=2)
        ))

        for house_id, house_data in consumption_data.items():
            for appliance_id, data in house_data.items():
                hourly_consumption_list = [data['hourly_consumption'].get(i, 0) / days for i in range(24)]
                fig.add_trace(go.Scatter(
                    name=f"House {house_id} - {data['name']} (ID: {appliance_id})",
                    x=hours,
                    y=hourly_consumption_list,
                    mode='lines'
                ))

        fig.update_layout(
            title=f'Average Daily Consumption by Appliance (Houses: {", ".join(map(str, selected_houses))})',
            xaxis_title='Hour of the Day',
            yaxis_title='Consumption (kWh)',
            xaxis=dict(tickmode='linear', tick0=0, dtick=1)
        )

        fig.show()

        # Total consumption by appliance plot
        fig_total = go.Figure()

        for house_id, house_data in consumption_data.items():
            fig_total.add_trace(go.Bar(
                name=f'House {house_id}',
                x=[f"{data['name']} (ID: {appliance_id})" for appliance_id, data in house_data.items()],
                y=[data['total_consumption'] for data in house_data.values()]
            ))

        fig_total.update_layout(
            title=f'Annual Consumption by Appliance (Houses: {", ".join(map(str, selected_houses))})',
            xaxis_title='Appliance',
            yaxis_title='Total Consumption (kWh)',
            xaxis_tickangle=-45,
            barmode='group'
        )

        fig_total.show()

        

    def analyze_houses(self, selected_houses):
        if not selected_houses:
            print("Please select at least one house.")
            return
        
        consumption_data = {}
        for house_id in selected_houses:
            try:
                consumption_data[house_id] = self.calculate_appliance_consumption(house_id)
            except Exception as e:
                print(f"An error occurred for House ID {house_id}: {str(e)}")
        
        if consumption_data:
            self.plot_consumption(selected_houses, consumption_data)